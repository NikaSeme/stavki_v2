"""
League Weights Optimizer
========================

Optimizes ensemble weights for each league individually using Log Loss.
Saves results to models/league_weights.json.

Usage:
    python scripts/optimize_league_weights.py
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import log_loss

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from stavki.models.ensemble.predictor import EnsemblePredictor, Market
from stavki.models.base import BaseModel

def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare data for optimization."""
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
    # Temporal Split - We optimize on VAL set (60-80%)
    # Train: 0-60%
    # Val: 60-80% (Optimization Target)
    # Test: 80-100% (Final Evaluation)
    
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    logger.info(f"Loaded validation set: {len(val_df)} matches")
    
    # Ensure target exists
    if "FTR" in val_df.columns:
        val_df["target"] = val_df["FTR"].map({"H": "home", "D": "draw", "A": "away"})
        
    # Ensure match_id exists and is standardized
    from stavki.utils import generate_match_id
    val_df["match_id"] = val_df.apply(
        lambda x: generate_match_id(x.get("HomeTeam", ""), x.get("AwayTeam", ""), x.get("Date")), 
        axis=1
    )
    
    return val_df

def load_models(models_dir: Path) -> EnsemblePredictor:
    """Load all trained models into an ensemble."""
    ensemble = EnsemblePredictor()
    
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
    if not model_files:
        logger.error(f"No models found in {models_dir}")
        sys.exit(1)
        
    for path in model_files:
        try:
            # Skip ensemble/calibrator files
            if "calibrator" in path.name or "ensemble" in path.name:
                continue
                
            model = BaseModel.load(path)
            ensemble.add_model(model)
            logger.info(f"Loaded {model.name}")
        except Exception as e:
            logger.warning(f"Failed to load {path.name}: {e}")
            
    if not ensemble.models:
        logger.error("No valid models loaded")
        sys.exit(1)
        
    return ensemble

def optimize_weights(preds: dict, actuals: pd.Series) -> dict:
    """
    Optimize weights for a set of predictions using Log Loss.
    preds: {model_name: np.array(n_samples, 3)}
    actuals: pd.Series of 'home', 'draw', 'away'
    """
    models = list(preds.keys())
    n_models = len(models)
    if n_models < 2:
        return {m: 1.0 for m in models}
    
    # Map targets to indices
    y_true = actuals.map({"home": 0, "draw": 1, "away": 2}).values
    
    # Stack predictions: (n_samples, n_models, 3)
    # Actually we just need to mix them
    
    def loss_func(weights):
        # Normalize weights
        w = np.abs(weights)
        w = w / w.sum()
        
        # Weighted sum of probs
        final_probs = np.zeros_like(preds[models[0]])
        for i, m in enumerate(models):
            final_probs += w[i] * preds[m]
            
        # Clip
        final_probs = np.clip(final_probs, 1e-15, 1 - 1e-15)
        # Normalize (just in case)
        final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)
        
        return log_loss(y_true, final_probs)
    
    # Initial weights
    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    res = minimize(loss_func, init_w, bounds=bounds, constraints=constraints, method='SLSQP')
    
    if res.success:
        best_w = np.abs(res.x) / np.sum(np.abs(res.x))
        # Zero out small weights
        best_w[best_w < 0.01] = 0
        best_w = best_w / best_w.sum()
        return {m: float(w) for m, w in zip(models, best_w)}
    else:
        logger.warning("Optimization failed, using equal weights")
        return {m: 1.0/n_models for m in models}

def main():
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    val_df = load_data(data_path)
    
    models_dir = PROJECT_ROOT / "models"
    ensemble = load_models(models_dir)
    
    # Filter valid leagues (at least 50 matches)
    league_counts = val_df["League"].value_counts()
    valid_leagues = league_counts[league_counts >= 50].index.tolist()
    
    logger.info(f"Optimizing for {len(valid_leagues)} leagues: {valid_leagues}")
    
    # Global component predictions
    # Cache predictions to avoid re-running
    logger.info("Generating predictions...")
    model_preds_cache = {}
    
    # We only care about MATCH_WINNER (1x2) for now as it's the primary market
    # Could extend to BTTS/OU later if models support it
    
    for name, model in ensemble.models.items():
        if not model.supports_market(Market.MATCH_WINNER):
            continue
            
        try:
            # Predict on full val_df
            preds = model.predict(val_df)
            
            # Extract 1x2 probs
            # Faster: Create dict lookup
            pred_map = {str(p.match_id): p for p in preds if p.market == Market.MATCH_WINNER}
            
            model_preds_cache[name] = pred_map
        except Exception as e:
            logger.warning(f"{name} failed prediction: {e}")

    league_weights = {}
    
    # Per-League Optimization
    for league in valid_leagues:
        league_df = val_df[val_df["League"] == league]
        actuals = league_df["target"].dropna()
        
        if len(actuals) < 50:
            continue
            
        # Collect aligned predictions for this league
        league_preds = {}
        valid_idx = []
        
        for idx, row in league_df.iterrows():
            # Try both explicit ID and generated ID
            possible_ids = []
            if "match_id" in row:
                possible_ids.append(str(row["match_id"]))
            
            from stavki.utils import generate_match_id
            gen_id = generate_match_id(row["HomeTeam"], row["AwayTeam"], row["Date"])
            possible_ids.append(gen_id)
            
            # Check availability
            row_preds = {}
            has_all = True
            
            for m_name, p_map in model_preds_cache.items():
                found_pred = None
                for mid in possible_ids:
                    if mid in p_map:
                        found_pred = p_map[mid]
                        break
                
                if found_pred:
                    p = found_pred.probabilities
                    row_preds[m_name] = np.array([p.get("home", 0), p.get("draw", 0), p.get("away", 0)])
                else:
                    has_all = False
                    break
            
            if has_all:
                for m_name, arr in row_preds.items():
                    if m_name not in league_preds:
                        league_preds[m_name] = []
                    league_preds[m_name].append(arr)
                
                valid_idx.append(row["target"])

        if not valid_idx or len(valid_idx) < 20:
            logger.warning(f"Skipping {league}: insufficient aligned predictions ({len(valid_idx)})")
            continue
            
        # Convert to arrays
        formatted_preds = {m: np.array(arr) for m, arr in league_preds.items()}
        formatted_actuals = pd.Series(valid_idx)
        
        logger.info(f"Optimizing {league} ({len(formatted_actuals)} matches)...")
        weights = optimize_weights(formatted_preds, formatted_actuals)
        
        # Store
        # Structure: {"1x2": {model: weight}}
        league_weights[league] = {"1x2": weights}
        
        logger.info(f"  -> {weights}")

    # Save
    output_path = models_dir / "league_weights.json"
    with open(output_path, "w") as f:
        json.dump(league_weights, f, indent=2)
        
    logger.info(f"Saved league weights to {output_path}")

if __name__ == "__main__":
    main()
