#!/usr/bin/env python3
"""
Ensemble Model Evaluation
=========================

Trains each component model (Poisson, CatBoost, LightGBM) on historical data
and evaluates accuracy per league + overall. Then builds a weighted ensemble
and reports combined performance.

Usage:
    python3 scripts/evaluate_ensemble.py
    python3 scripts/evaluate_ensemble.py --data data/features_full.csv
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# League display names
LEAGUE_NAMES = {
    "E0": "EPL",
    "E1": "Championship",
    "D1": "Bundesliga",
    "SP1": "La Liga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}


def load_data(path: Path) -> pd.DataFrame:
    """Load and validate features dataset."""
    df = pd.read_csv(path, low_memory=False)
    
    required = ["HomeTeam", "AwayTeam", "FTR"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Clean FTR
    df = df[df["FTR"].isin(["H", "D", "A"])].copy()
    df["FTR_num"] = df["FTR"].map({"H": 0, "D": 1, "A": 2})
    
    # Identify league column
    if "Div" in df.columns:
        df["league"] = df["Div"]
    elif "League" in df.columns:
        df["league"] = df["League"]
    else:
        df["league"] = "unknown"
    
    logger.info(f"Loaded {len(df)} matches, {df['league'].nunique()} leagues")
    return df


def build_features(df: pd.DataFrame) -> tuple:
    """Build feature matrix X and target y from DataFrame."""
    skip_cols = {
        "HomeTeam", "AwayTeam", "Date", "Time", "FTR", "FTR_num",
        "Result", "Season", "League", "Div", "Referee", "league",
        "HTR",  # Half-time result (leakage)
    }
    
    feature_cols = [
        c for c in df.columns
        if c not in skip_cols and df[c].dtype in [np.int64, np.float64, float, int]
    ]
    
    # Remove half-time and full-time goal columns (potential leakage)
    leakage_cols = {"FTHG", "FTAG", "HTHG", "HTAG", "HS", "AS", "HST", "AST",
                    "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR"}
    feature_cols = [c for c in feature_cols if c not in leakage_cols]
    
    X = df[feature_cols].fillna(0)
    y = df["FTR_num"]
    
    logger.info(f"Features: {len(feature_cols)} columns")
    return X, y, feature_cols


def temporal_split(df, X, y, test_frac=0.20, val_frac=0.10):
    """Temporal split: train -> val -> test (no shuffling)."""
    n = len(df)
    train_end = int(n * (1 - test_frac - val_frac))
    val_end = int(n * (1 - test_frac))
    
    return {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_val": X.iloc[train_end:val_end],
        "y_val": y.iloc[train_end:val_end],
        "X_test": X.iloc[val_end:],
        "y_test": y.iloc[val_end:],
        "df_train": df.iloc[:train_end],
        "df_val": df.iloc[train_end:val_end],
        "df_test": df.iloc[val_end:],
    }


def compute_metrics(y_true, y_pred_proba, y_pred_class):
    """Compute accuracy, log loss, and Brier score."""
    accuracy = (y_pred_class == y_true).mean()
    
    # Log loss
    eps = 1e-10
    log_loss = 0
    for i, true_cls in enumerate(y_true):
        if 0 <= true_cls < y_pred_proba.shape[1]:
            prob = max(y_pred_proba[i, int(true_cls)], eps)
            log_loss -= np.log(prob)
    log_loss /= max(len(y_true), 1)
    
    # Brier score (multiclass)
    brier = 0
    for i, true_cls in enumerate(y_true):
        for c in range(y_pred_proba.shape[1]):
            target = 1.0 if c == true_cls else 0.0
            brier += (y_pred_proba[i, c] - target) ** 2
    brier /= max(len(y_true), 1)
    
    return {"accuracy": accuracy, "log_loss": log_loss, "brier": brier}


def evaluate_per_league(y_true, y_pred_proba, y_pred_class, leagues):
    """Compute metrics per league."""
    results = {}
    for league in sorted(leagues.unique()):
        mask = leagues == league
        if mask.sum() < 10:
            continue
        
        yt = y_true[mask].values
        ypp = y_pred_proba[mask.values]
        ypc = y_pred_class[mask.values]
        
        metrics = compute_metrics(yt, ypp, ypc)
        metrics["n_matches"] = int(mask.sum())
        name = LEAGUE_NAMES.get(league, league)
        results[name] = metrics
    
    return results


# =====================================================
# Individual Model Trainers
# =====================================================

def train_poisson(data):
    """Train Dixon-Coles Poisson model."""
    logger.info("  Training Poisson (Dixon-Coles)...")
    start = time.time()
    
    try:
        from stavki.models.poisson import DixonColesModel
        from stavki.models.base import Market
        
        model = DixonColesModel()
        # Poisson model expects a DataFrame with HomeTeam, AwayTeam, FTHG, FTAG
        model.fit(data["df_train"])
        
        # Batch predict on DataFrame
        all_predictions = model.predict(data["df_test"])
        
        # Filter for 1x2 market only
        predictions_1x2 = [p for p in all_predictions if p.market == Market.MATCH_WINNER]
        
        if len(predictions_1x2) != len(data["df_test"]):
            logger.warning(f"Poisson returned {len(predictions_1x2)} 1x2 predictions for {len(data['df_test'])} rows")
            # If mismatch, fallback to row-by-row to ensure alignment
            predictions_1x2 = []
            for _, row in data["df_test"].iterrows():
                row_preds = model.predict(row.to_frame().T if hasattr(row, 'to_frame') else pd.DataFrame([row]))
                p_1x2 = next((p for p in row_preds if p.market == Market.MATCH_WINNER), None)
                if p_1x2:
                    predictions_1x2.append(p_1x2)
                else:
                    # Dummy
                    from stavki.models.base import Prediction
                    predictions_1x2.append(Prediction(
                        match_id="dummy", market=Market.MATCH_WINNER, probabilities={"home": 0.33, "draw": 0.33, "away": 0.34}
                    ))
        
        probs_list = []
        for p in predictions_1x2:
            probs = p.probabilities
            probs_list.append([probs.get("home", 0.33), probs.get("draw", 0.33), probs.get("away", 0.33)])
            
        y_pred_proba = np.array(probs_list)
        y_pred_class = y_pred_proba.argmax(axis=1)
        
        elapsed = time.time() - start
        logger.info(f"  Poisson trained in {elapsed:.1f}s")
        return y_pred_proba, y_pred_class, elapsed
        
    except Exception as e:
        logger.error(f"  Poisson failed: {e}")
        n = len(data["y_test"])
        return np.full((n, 3), 1/3), np.zeros(n, dtype=int), 0


def train_catboost(data):
    """Train CatBoost model."""
    logger.info("  Training CatBoost...")
    start = time.time()
    
    try:
        from stavki.models.catboost import CatBoostModel
        from stavki.models.base import Market
        
        model = CatBoostModel(iterations=100)
        
        # Check validation dataframe availability
        if "df_val" not in data:
             logger.error("df_val missing from data split")
             raise KeyError("df_val")
            
        # CatBoostModel.fit expects a single DataFrame and does internal splitting
        # So we reconstruct the Train + Val set
        train_val_df = pd.concat([data["df_train"], data["df_val"]])
        
        # Calculate eval_ratio so the split point matches our existing val set
        total_len = len(train_val_df)
        val_len = len(data["df_val"])
        eval_ratio = val_len / total_len if total_len > 0 else 0.1
        
        logger.info(f"  Fitting with n={total_len}, eval_ratio={eval_ratio:.3f}")
        
        model.fit(
            train_val_df,
            eval_ratio=eval_ratio,
            verbose=0
        )
        
        # CatBoostModel.predict takes a DataFrame
        all_predictions = model.predict(data["df_test"])
        
        # Filter for 1x2 market
        predictions_1x2 = [p for p in all_predictions if p.market == Market.MATCH_WINNER]
        
        probs_list = []
        for p in predictions_1x2:
            probs = p.probabilities
            probs_list.append([probs.get("home", 0.33), probs.get("draw", 0.33), probs.get("away", 0.33)])
            
        y_pred_proba = np.array(probs_list)
        y_pred_class = y_pred_proba.argmax(axis=1)
        
        elapsed = time.time() - start
        logger.info(f"  CatBoost trained in {elapsed:.1f}s")
        return y_pred_proba, y_pred_class, elapsed
        
    except Exception as e:
        logger.error(f"  CatBoost failed: {e}")
        import traceback
        traceback.print_exc()
        n = len(data["y_test"])
        return np.full((n, 3), 1/3), np.zeros(n, dtype=int), 0


def train_lightgbm(data):
    """Train LightGBM model."""
    logger.info("  Training LightGBM...")
    start = time.time()
    
    try:
        from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
        from stavki.models.base import Market
        
        feature_list = list(data["X_train"].columns)
        model = LightGBMModel(n_estimators=100, features=feature_list)
        
        if "df_val" not in data:
             logger.error("df_val missing from data split")
             raise KeyError("df_val")
        
        # Reconstruct Train + Val
        train_val_df = pd.concat([data["df_train"], data["df_val"]])
        
        total_len = len(train_val_df)
        val_len = len(data["df_val"])
        eval_ratio = val_len / total_len if total_len > 0 else 0.1
        
        model.fit(
            train_val_df,
            eval_ratio=eval_ratio,
        )
        
        all_predictions = model.predict(data["df_test"])
        
        # Filter for 1x2 market
        predictions_1x2 = [p for p in all_predictions if p.market == Market.MATCH_WINNER]
        
        probs_list = []
        for p in predictions_1x2:
            probs = p.probabilities
            probs_list.append([probs.get("home", 0.33), probs.get("draw", 0.33), probs.get("away", 0.33)])
            
        y_pred_proba = np.array(probs_list)
        y_pred_class = y_pred_proba.argmax(axis=1)
        
        elapsed = time.time() - start
        logger.info(f"  LightGBM trained in {elapsed:.1f}s")
        return y_pred_proba, y_pred_class, elapsed
        
    except Exception as e:
        logger.error(f"  LightGBM failed: {e}")
        import traceback
        traceback.print_exc()
        n = len(data["y_test"])
        return np.full((n, 3), 1/3), np.zeros(n, dtype=int), 0


def build_ensemble(model_outputs, weights=None):
    """Build weighted ensemble from individual model predictions."""
    names = list(model_outputs.keys())
    
    if weights is None:
        # Default equal weights
        weights = {name: 1.0 / len(names) for name in names}
    
    # Normalize weights
    total_w = sum(weights.get(n, 0) for n in names)
    norm_weights = {n: weights.get(n, 0) / total_w for n in names}
    
    # Weighted average of probability distributions
    n_test = model_outputs[names[0]][0].shape[0]
    ensemble_proba = np.zeros((n_test, 3))
    
    for name in names:
        y_proba, _, _ = model_outputs[name]
        ensemble_proba += norm_weights[name] * y_proba
    
    # Renormalize rows
    row_sums = ensemble_proba.sum(axis=1, keepdims=True)
    ensemble_proba = ensemble_proba / np.maximum(row_sums, 1e-10)
    
    ensemble_class = ensemble_proba.argmax(axis=1)
    return ensemble_proba, ensemble_class


def print_results_table(all_results):
    """Pretty-print results table."""
    # Header
    print("\n" + "=" * 90)
    print(f"{'Model':<15} {'League':<15} {'Matches':>8} {'Accuracy':>10} {'Log Loss':>10} {'Brier':>8}")
    print("-" * 90)
    
    for model_name, league_results in all_results.items():
        first = True
        for league, metrics in sorted(league_results.items()):
            name_col = model_name if first else ""
            print(
                f"{name_col:<15} {league:<15} "
                f"{metrics['n_matches']:>8} "
                f"{metrics['accuracy']:>9.1%} "
                f"{metrics['log_loss']:>10.4f} "
                f"{metrics['brier']:>8.4f}"
            )
            first = False
        print("-" * 90)
    
    print("=" * 90)


def print_summary_table(summary):
    """Print overall summary across models."""
    print("\n" + "=" * 70)
    print(f"{'Model':<15} {'Overall Acc':>12} {'Log Loss':>10} {'Brier':>8} {'Time':>8}")
    print("-" * 70)
    
    for name, m in summary.items():
        print(
            f"{name:<15} "
            f"{m['accuracy']:>11.1%} "
            f"{m['log_loss']:>10.4f} "
            f"{m['brier']:>8.4f} "
            f"{m.get('time', 0):>7.1f}s"
        )
    
    print("=" * 70)
    
    # Highlight best
    best_acc = max(summary.items(), key=lambda x: x[1]["accuracy"])
    best_ll = min(summary.items(), key=lambda x: x[1]["log_loss"])
    print(f"\n🏆 Best accuracy: {best_acc[0]} ({best_acc[1]['accuracy']:.1%})")
    print(f"🏆 Best log loss: {best_ll[0]} ({best_ll[1]['log_loss']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate STAVKI ensemble models")
    parser.add_argument("--data", default="data/features_full.csv", help="Path to features CSV")
    parser.add_argument("--test-frac", type=float, default=0.20, help="Test set fraction")
    args = parser.parse_args()
    
    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
    
    # 1. Load and prepare data
    print("\n📊 STAVKI Ensemble Evaluation")
    print("=" * 50)
    
    df = load_data(data_path)
    X, y, feature_cols = build_features(df)
    data = temporal_split(df, X, y, test_frac=args.test_frac)
    
    print(f"Train: {len(data['X_train'])} | Val: {len(data['X_val'])} | Test: {len(data['X_test'])}")
    print(f"Test leagues: {data['df_test']['league'].value_counts().to_dict()}")
    
    # 2. Train individual models
    print("\n🧠 Training Models...")
    print("-" * 50)
    
    model_outputs = {}
    trainers = {
        "Poisson": train_poisson,
        "CatBoost": train_catboost,
        "LightGBM": train_lightgbm,
    }
    
    for name, trainer in trainers.items():
        model_outputs[name] = trainer(data)
    
    # 3. Build ensemble
    print("\n🔗 Building Ensemble...")
    
    # Weighted ensemble (CatBoost-heavy, reflects typical production config)
    ensemble_weights = {"Poisson": 0.25, "CatBoost": 0.45, "LightGBM": 0.30}
    ens_proba, ens_class = build_ensemble(model_outputs, weights=ensemble_weights)
    model_outputs["Ensemble"] = (ens_proba, ens_class, 0)
    
    # 4. Evaluate all models
    print("\n📈 Evaluation Results")
    
    all_league_results = {}
    summary = {}
    
    test_leagues = data["df_test"]["league"].reset_index(drop=True)
    y_test = data["y_test"].reset_index(drop=True)
    
    for name, (y_proba, y_class, elapsed) in model_outputs.items():
        # Overall
        overall = compute_metrics(y_test.values, y_proba, y_class)
        overall["n_matches"] = len(y_test)
        overall["time"] = elapsed
        summary[name] = overall
        
        # Per league
        league_results = evaluate_per_league(y_test, y_proba, y_class, test_leagues)
        league_results["OVERALL"] = overall
        all_league_results[name] = league_results
    
    # 5. Print results
    print_results_table(all_league_results)
    print_summary_table(summary)
    
    # 6. Baseline comparison
    # Most common class (Home win) baseline
    baseline_acc = (y_test == 0).mean()  # 0 = Home
    print(f"\n📌 Baseline (always Home): {baseline_acc:.1%}")
    
    # Market implied baseline (if odds columns exist)
    if "B365H" in data["df_test"].columns:
        test_df = data["df_test"].reset_index(drop=True)
        market_pred = []
        for _, row in test_df.iterrows():
            odds = [row.get("B365H", 99), row.get("B365D", 99), row.get("B365A", 99)]
            market_pred.append(np.argmin(odds))  # Lowest odds = favorite
        market_pred = np.array(market_pred)
        market_acc = (market_pred == y_test.values).mean()
        print(f"📌 Market favorite: {market_acc:.1%}")
    
    print("\nDone! ✅\n")


if __name__ == "__main__":
    main()
