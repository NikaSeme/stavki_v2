"""
Ensemble Weights Optimizer (ROI & Kelly Criterion)
==================================================

Optimizes ensemble blending weights to maximize Return on Investment (ROI)
rather than minimizing Log Loss. This directly aligns model predictions
with the actual Kelly staking logic used by the bot in production.

Usage:
    python scripts/optimize_ensemble_kelly.py
"""

import sys
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from stavki.models.ensemble.predictor import EnsemblePredictor, Market
from stavki.models.base import BaseModel
from stavki.utils import generate_match_id

# ── 1. Configuration ──
MIN_LEAGUE_MATCHES = 100
MIN_PROB = 0.40       # Minimum model prediction confidence
MIN_EV = 0.05         # Minimum Expected Value (5% edge) to place a bet
MAX_STAKE = 50.0      # Cap Kelly stakes

def load_data(data_path: Path) -> pd.DataFrame:
    """Load and chronologically split the full features data."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        
    # We optimize on the Validation Set (60-80% of time)
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    logger.info(f"Loaded Validation Set: {len(val_df)} matches spanning {val_df['Date'].min().date()} to {val_df['Date'].max().date()}")
    
    # Standardize Targets
    if "FTR" in val_df.columns:
        val_df["target"] = val_df["FTR"].map({"H": "home", "D": "draw", "A": "away"})
        
    return val_df

def load_ensemble(models_dir: Path) -> EnsemblePredictor:
    """Load all valid models from the models directory."""
    logger.info("Loading models into ensemble...")
    ensemble = EnsemblePredictor(use_disagreement=False) 
    
    model_files = list(models_dir.glob("*.pkl")) + list(models_dir.glob("*.joblib"))
    
    # 1. Load Sklearn/CatBoost models
    for path in model_files:
        try:
            if "calibrator" in path.name or "ensemble" in path.name:
                continue
                
            model = BaseModel.load(path)
            if model.supports_market(Market.MATCH_WINNER):
                ensemble.add_model(model)
        except Exception as e:
            logger.warning(f"Failed to load {path.name}: {e}")
            
    # 2. Load PyTorch model explicitly
    import torch
    from stavki.models.deep_interaction_wrapper import DeepInteractionWrapper
    deep_path = models_dir / "deep_interaction_v3.pth"
    if deep_path.exists():
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            deep_model = DeepInteractionWrapper()
            deep_model.device = device # Set directly
            if deep_model.load_checkpoint(deep_path):
                ensemble.add_model(deep_model)
                logger.info(f"Loaded DeepInteraction Network from {deep_path.name}")
        except Exception as e:
            logger.error(f"Failed to load DeepInteraction Network: {e}")
            
    if not ensemble.models:
        logger.error("No MATCH_WINNER models loaded.")
        sys.exit(1)
        
    return ensemble

def map_predictions(models, val_df, preds_cache):
    """
    Extracts the (N, M, 3) tensor of predictions where:
      - N is the number of valid matches that all models predicted.
      - M is the model count.
      - 3 are the (Home, Draw, Away) probabilities.
    Returns the aligned predictions tensor, actual target strings, and respective odds.
    """
    aligned_preds = []
    actuals = []
    odds_data = [] # List of dicts {"home": B365H, "draw": B365D, "away": B365A}
    
    # Pre-cache available match IDs for O(1) lookups
    match_ids = val_df.apply(lambda x: generate_match_id(x.get("HomeTeam", ""), x.get("AwayTeam", ""), x.get("Date")), axis=1)
    
    valid_count = 0
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Aligning Predictions"):
        mid = match_ids.iloc[idx]
        target = row.get("target")
        
        # We need actual results and odds to calculate ROI
        if pd.isna(target):
            continue
            
        home_odds = row.get("B365H", row.get("PSH", row.get("BWH", 0)))
        draw_odds = row.get("B365D", row.get("PSD", row.get("BWD", 0)))
        away_odds = row.get("B365A", row.get("PSA", row.get("BWA", 0)))
        
        if any(np.isnan([home_odds, draw_odds, away_odds])) or any(o <= 1.0 for o in [home_odds, draw_odds, away_odds]):
            continue
            
        row_preds = []
        has_all_predictions = True
        
        # Fallback to idx generation for CatBoost
        catboost_mid = f"{row.get('HomeTeam', '')}_vs_{row.get('AwayTeam', '')}_{idx}"
        
        for m_name in models:
            pred_map = preds_cache.get(m_name, {})
            # Look up prediction object natively
            p = pred_map.get(mid)
            if p is None:
                # Fallback check
                p = pred_map.get(catboost_mid)
                if p is None:
                    # Case insensitive fallback
                    lower_keys = {k.lower(): v for k, v in pred_map.items()}
                    p = lower_keys.get(catboost_mid.lower())
                    if p is None:
                        has_all_predictions = False
                        break
                
            probs = p.probabilities
            row_preds.append([probs.get("home", 0), probs.get("draw", 0), probs.get("away", 0)])
            
        if has_all_predictions:
            aligned_preds.append(row_preds)
            actuals.append(target)
            odds_data.append({"home": home_odds, "draw": draw_odds, "away": away_odds})
            valid_count += 1
            
    return np.array(aligned_preds), actuals, odds_data

def optimize_roi_weights(preds_tensor, actuals, odds_data, model_names):
    """
    Simulates betting using fractional Kelly criteria to find weighting
    coefficients that maximize total Net Profit.
    """
    n_models = len(model_names)
    n_samples = len(actuals)
    
    outcome_map = {"home": 0, "draw": 1, "away": 2}
    
    def kelly_objective(weights):
        # 1. Normalize weights
        w = np.abs(weights)
        w_sum = w.sum()
        if w_sum == 0:
            return 10000.0 # Heavy penalty
        w = w / w_sum
        
        # 2. Blend Predictions (N_samples, 3)
        # preds_tensor is (N, M, 3)
        # We dot product the M dimension with the Weights vector
        blended_probs = np.tensordot(preds_tensor, w, axes=([1], [0]))
        
        total_profit = 0.0
        
        # 3. Betting Simulation Loop
        for i in range(n_samples):
            probs = blended_probs[i]
            row_odds = odds_data[i]
            actual_res = actuals[i]
            
            # Find best value bet 
            best_idx = np.argmax(probs)
            keys = ["home", "draw", "away"]
            pred_outcome = keys[best_idx]
            model_p = probs[best_idx]
            
            decimal_odds = row_odds[pred_outcome]
            ev = (model_p * decimal_odds) - 1.0
            
            # Constraints
            if model_p < MIN_PROB or ev < MIN_EV:
                continue
                
            # Kelly Criterion: Stake = (bp - q) / b  = EV / (Odds - 1)
            # Fractional Kelly (Quarter Kelly) for safety
            b = decimal_odds - 1.0
            q = 1.0 - model_p
            f_star = ((model_p * b) - q) / b
            
            if f_star <= 0:
                continue
                
            # Scale Stake -> 1000 unit bankroll assumption
            stake = min(f_star * 0.25 * 1000, MAX_STAKE)
            
            if pred_outcome == actual_res:
                total_profit += stake * b
            else:
                total_profit -= stake
                
        # Scipy minimize looks for the lowest number. 
        # We want Highest Profit, so we return negative profit.
        return -total_profit

    # Initial Guesses - Equal weight
    init_w = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    
    logger.info("Executing SciPy SLSQP minimizer...")
    res = minimize(
        kelly_objective, 
        init_w, 
        bounds=bounds, 
        constraints=constraints, 
        method='SLSQP',
        options={'maxiter': 200, 'ftol': 1e-4}
    )
    
    if res.success:
        best_w = np.abs(res.x)
        best_w = best_w / best_w.sum()
        
        # Evaluate Best Run
        max_profit = -res.fun
        
        # Zero out negligible weights
        best_w[best_w < 0.01] = 0
        best_w = best_w / best_w.sum()
        
        logger.info(f"Optimization Succeeded! Max Val Profit: +{max_profit:.2f} Units")
        return {m: float(w) for m, w in zip(model_names, best_w)}
        
    else:
        logger.warning(f"Optimization failed: {res.message}")
        return {m: 1.0 / n_models for m in model_names}

def main():
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    val_df = load_data(data_path)
    
    models_dir = PROJECT_ROOT / "models"
    ensemble = load_ensemble(models_dir)
    models = list(ensemble.models.keys())
    
    logger.info("Generating internal model predictions on the Validation Set...")
    model_preds_cache = {}
            
    for name, model in ensemble.models.items():
        # For DeepInteraction Network and CatBoost
        preds = model.predict(val_df)
        if preds is None:
            logger.error(f"  -> {name} returned None instead of a prediction list!")
            continue
        pred_map = {str(p.match_id): p for p in preds if p.market == Market.MATCH_WINNER}
        model_preds_cache[name] = pred_map
        logger.info(f"  -> Generated {len(pred_map)} predictions for {name}")
        
    models = list(model_preds_cache.keys())
            
    # ── ALIGNMENT ──
    logger.info("Aligning predictions to find consensus intersections...")
    preds_tensor, actuals, odds_data = map_predictions(models, val_df, model_preds_cache)
    
    if len(actuals) < 500:
        logger.error(f"Insufficient aligned matches ({len(actuals)}). Aborting optimization.")
        logger.info("Dumping first 2 prediction IDs from active models for debug intersection:")
        for name, p_map in model_preds_cache.items():
            sample_ids = list(p_map.keys())[:2]
            logger.info(f"Model {name} sample match_ids: {sample_ids}")
        sys.exit(1)
        
    logger.info(f"Successfully aligned {len(actuals)} matches across {len(models)} models.")
    
    # ── OPTIMIZATION ──
    weights = optimize_roi_weights(preds_tensor, actuals, odds_data, models)
    
    logger.info("\n🏆 Optimal Ensemble Weights (Kelly Calibrated) 🏆")
    for m, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {m:<20}: {w:.4f}")
        
    # Overwrite the league configs
    output_path = models_dir / "roi_weights.json"
    weighted_market = {"MATCH_WINNER": weights}
    
    with open(output_path, "w") as f:
        json.dump(weighted_market, f, indent=4)
        
    logger.info(f"Weights saved globally to {output_path}")

if __name__ == "__main__":
    main()
