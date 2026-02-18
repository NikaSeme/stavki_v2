"""
Fit Calibrator — Train isotonic regression calibrators on historical data.

Reads the historical feature set, generates predictions from the ensemble,
and fits per-outcome calibrators using a temporal split (most recent 20% as
validation). Supports all three markets: 1X2, BTTS, Over/Under.

Output:
    models/calibrator.joblib — serialized EnsembleCalibrator

Usage:
    python scripts/fit_calibrator.py [--method isotonic|platt] [--val-ratio 0.2]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
import joblib
import numpy as np
import pandas as pd

from stavki.models.ensemble.predictor import EnsemblePredictor
from stavki.models.ensemble.calibrator import EnsembleCalibrator
from stavki.models.base import Market, Prediction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def load_historical(csv_path: Path) -> pd.DataFrame:
    """Load and validate historical data."""
    df = pd.read_csv(csv_path)
    required = ["HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["match_id"] = df.index.astype(str)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} historical matches from {csv_path.name}")
    return df


def temporal_split(df: pd.DataFrame, val_ratio: float = 0.2):
    """
    Split into fit/calibrate sets.
    
    CRITICAL: This must align with retrain_system.py's split!
    retrain_system.py usage:
      - Train: 0-60%
      - Val (Early Stop): 60-80%
      - Test (Held Out): 80-100%
      
    Here, we want to calibrate ON THE HELD OUT TEST SET (80-100%).
    So we take the last 20% (val_ratio=0.2).
    """
    split_idx = int(len(df) * (1 - val_ratio))
    train = df.iloc[:split_idx].reset_index(drop=True) # Used for generating predictions (0-80%)
    val = df.iloc[split_idx:].reset_index(drop=True)   # Used for fitting calibrator (80-100%)
    logger.info(f"Split: {len(train)} predictor-fit / {len(val)} calibration-fit (ratio={val_ratio})")
    return train, val


def predict_batch(
    ensemble: EnsemblePredictor, df: pd.DataFrame, batch_size: int = 500,
) -> list[Prediction]:
    """Run ensemble prediction in batches to manage memory."""
    all_preds = []
    n_batches = (len(df) + batch_size - 1) // batch_size
    for i in range(n_batches):
        chunk = df.iloc[i * batch_size : (i + 1) * batch_size]
        try:
            preds = ensemble.predict(chunk)
            all_preds.extend(preds)
        except Exception as e:
            logger.warning(f"Batch {i+1}/{n_batches} failed: {e}")
    logger.info(f"Predicted {len(all_preds)} total predictions across {n_batches} batches")
    return all_preds


def build_actuals(df: pd.DataFrame) -> dict[str, dict[str, str]]:
    """
    Build per-market ground truth for each match.
    
    Returns:
        {match_id: {"1x2": "home"|"draw"|"away",
                     "btts": "yes"|"no",  
                     "over_under": "over_2.5"|"under_2.5"}}
    """
    FTR_MAP = {"H": "home", "D": "draw", "A": "away"}
    
    actuals = {}
    for _, row in df.iterrows():
        mid = str(row["match_id"])
        fthg = row.get("FTHG", 0)
        ftag = row.get("FTAG", 0)
        ftr = row.get("FTR", "")
        
        match_actuals = {}
        
        # 1X2
        outcome_1x2 = FTR_MAP.get(ftr)
        if outcome_1x2:
            match_actuals["1x2"] = outcome_1x2
        
        # BTTS
        if pd.notna(fthg) and pd.notna(ftag):
            btts = "yes" if (fthg > 0 and ftag > 0) else "no"
            match_actuals["btts"] = btts
        
        # Over/Under 2.5
        if pd.notna(fthg) and pd.notna(ftag):
            total = fthg + ftag
            ou = "over_2.5" if total > 2.5 else "under_2.5"
            match_actuals["over_under"] = ou
        
        if match_actuals:
            actuals[mid] = match_actuals
    
    return actuals


def main():
    parser = argparse.ArgumentParser(description="Fit probability calibrator")
    parser.add_argument(
        "--method", choices=["isotonic", "platt"], default="isotonic",
        help="Calibration method (default: isotonic)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Fraction of data to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--data", type=str, default="data/features_full.csv",
        help="Path to historical feature CSV",
    )
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    # 1. Load data
    df = load_historical(data_path)
    train_df, val_df = temporal_split(df, args.val_ratio)

    # 2. Load ensemble
    logger.info("Loading ensemble models...")
    from stavki.pipelines.daily import DailyPipeline
    pipeline = DailyPipeline()
    pipeline._init_components()
    ensemble = pipeline._load_ensemble()

    if not ensemble or not ensemble.models:
        logger.error("No ensemble models loaded. Train models first.")
        sys.exit(1)

    logger.info(f"Ensemble has {len(ensemble.models)} models")

    # 3. Generate predictions on validation set
    logger.info("Generating predictions on validation data...")
    val_preds = predict_batch(ensemble, val_df)

    if not val_preds:
        logger.error("No predictions generated. Check model compatibility.")
        sys.exit(1)

    # Count predictions per market
    market_counts = {}
    for p in val_preds:
        mk = p.market.value if hasattr(p.market, 'value') else str(p.market)
        market_counts[mk] = market_counts.get(mk, 0) + 1
    logger.info(f"Predictions by market: {market_counts}")

    # 4. Build per-market actuals
    actuals = build_actuals(val_df)
    logger.info(f"Built actuals for {len(actuals)} matches (all 3 markets)")

    # Sample to verify
    sample_id = list(actuals.keys())[0]
    logger.info(f"  Sample: match {sample_id} → {actuals[sample_id]}")

    # 5. Fit calibrator
    logger.info(f"Fitting calibrator (method={args.method})...")
    calibrator = EnsembleCalibrator(method=args.method)
    calibrator.fit(val_preds, actuals)

    n_calibrators = len(calibrator.calibrators)
    logger.info(f"Fitted {n_calibrators} outcome calibrators")
    logger.info(f"  Calibrator keys: {sorted(calibrator.calibrators.keys())}")

    if n_calibrators == 0:
        logger.warning(
            "No calibrators were fitted — insufficient data for any outcome. "
            "Check that predictions and actuals are aligned."
        )
        sys.exit(1)

    # 6. Evaluate calibration (ECE before and after)
    logger.info("\n" + "=" * 60)
    logger.info("CALIBRATION EVALUATION")
    logger.info("=" * 60)

    ece_before = calibrator.get_calibration_error(val_preds, actuals)
    calibrated_preds = calibrator.calibrate(val_preds)
    ece_after = calibrator.get_calibration_error(calibrated_preds, actuals)

    all_keys = sorted(set(ece_before) | set(ece_after))
    logger.info(f"\n{'Outcome':<25} {'ECE Before':>12} {'ECE After':>12} {'Δ':>12}")
    logger.info("-" * 65)
    for key in all_keys:
        before = ece_before.get(key, float("nan"))
        after = ece_after.get(key, float("nan"))
        delta = after - before if not (np.isnan(before) or np.isnan(after)) else float("nan")
        marker = " ✅" if delta < 0 else " ⚠️" if delta > 0 else ""
        logger.info(f"  {key:<23} {before:>11.4f}  {after:>11.4f}  {delta:>+11.4f}{marker}")

    # 7. Save
    output_path = PROJECT_ROOT / "models" / "calibrator.joblib"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, output_path)
    logger.info(f"\n✅ Calibrator saved to {output_path}")


if __name__ == "__main__":
    main()
