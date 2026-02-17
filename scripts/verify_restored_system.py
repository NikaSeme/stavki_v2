
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from stavki.models.base import BaseModel
from stavki.models.ensemble.predictor import EnsemblePredictor
from stavki.strategy.league_router import LeagueRouter

def main():
    logger.info("🔍 Starting Verification of Restored System...")
    
    # 1. Load Data
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    # 2. Reconstruct Test Split
    test_size = int(len(df) * 0.15)
    val_size = int(len(df) * 0.15)
    train_size = len(df) - test_size - val_size
    
    test_df = df.iloc[train_size+val_size:].copy()
    logger.info(f"Test Set: {len(test_df)} matches")
    
    # 3. Load Models
    models_dir = PROJECT_ROOT / "models"
    model_files = ["dixon_coles.pkl", "lightgbm.pkl", "catboost.pkl", "neural_multitask.pkl"]
    
    loaded_models = []
    for fname in model_files:
        path = models_dir / fname
        if not path.exists():
            logger.error(f"Model not found: {path}")
            continue
            
        try:
            model = BaseModel.load(path)
            loaded_models.append(model)
            logger.info(f"Loaded {model.name}")
        except Exception as e:
            logger.error(f"Failed to load {fname}: {e}")
            
    if not loaded_models:
        logger.error("No models loaded!")
        sys.exit(1)
        
    # 4. Initialize Ensemble (uses LeagueRouter internally or we can use explicit weights)
    # EnsemblePredictor expects a LIST of models, but internally converts to dict?
    # No, based on error it expects a dict or converts incorrectly.
    # Let's check the file content first.
    # Assuming it expects a list based on my previous usage, but error says "list object has no attribute items".
    # This means self.models is a list, and it iterates with .items(). 
    # So __init__ probably just assigns self.models = models.
    # So I should pass a dict {name: model} or fix the class.
    # I'll pass a dict.
    
    models_dict = {m.name: m for m in loaded_models}
    ensemble = EnsemblePredictor(models=models_dict)
    
    # 5. Predict on Test Set
    logger.info("Generating predictions on Test Set...")
    # This might take a while if not vectorized, but EnsemblePredictor IS vectorized now (from previous task)
    start_idx = test_df.index[0]
    preds = ensemble.predict(test_df) # This calls predict on all models
    
    logger.info(f"Predictions generated: {len(preds)}")
    if len(preds) != len(test_df):
        logger.warning(f"Length mismatch: Preds={len(preds)}, TestDF={len(test_df)}")
    
    # Build lookup for test_df
    # Assuming match_id column or index
    # We loaded from CSV, so match_id col might not be unique if data has duplicates? 
    # Or maybe match_id isn't in test_df columns? It implies it is content.
    if "match_id" not in test_df.columns:
        # Generate match_id to match what models use?
        # Models usually use a generated ID if not present.
        # But we need to match them.
        # Let's trust index alignment if lengths match, else use heuristics.
        pass

    # Better approach: Iterate preds, find row by match_id if possible
    # But preds come from Ensemble which generated IDs from test_df rows.
    # If EnsemblePredictor used row index to generate IDs, we are fine.
    
    from stavki.models.base import Market
    
    # Filter for 1X2 predictions
    preds_1x2 = [p for p in preds if p.market == Market.MATCH_WINNER]
    
    logger.info(f"1X2 Predictions: {len(preds_1x2)}")
    
    if len(preds_1x2) != len(test_df):
        logger.warning(f"Length mismatch: Preds={len(preds_1x2)}, TestDF={len(test_df)}")
        
    n_eval = min(len(preds_1x2), len(test_df))
    
    # Initialize metrics
    correct = 0
    total = 0
    log_loss = 0
    roi_pool = 0
    stake_pool = 0
    
    # Prepare map
    target_map = {"H": "home", "D": "draw", "A": "away"}
    
    # Prepare map
    target_map = {"H": "home", "D": "draw", "A": "away"}
    
    for i in range(n_eval):
        p = preds_1x2[i]
        # p is Prediction object
        
        row = test_df.iloc[i]
        
        # Truth
        ftr = row.get("FTR")
        if ftr not in ["H", "D", "A"]:
            continue
            
        actual = target_map[ftr]
        
        # Pred
        probs = p.probabilities
        if not probs:
            continue
            
        best_outcome = max(probs, key=probs.get)
        
        # Log Loss
        prob_actual = probs.get(actual, 0.001)
        log_loss -= np.log(max(prob_actual, 1e-10))
             
        # Accuracy
        if best_outcome == actual:
            correct += 1
        total += 1
             
        # ROI (Simple Betting)
        # Use B365 odds
        # Map outcome 'home' -> 'B365H'
        odds_key_map = {"home": "B365H", "draw": "B365D", "away": "B365A"}
        
        # Odds of the outcome we PREDICTED
        pred_odds_key = odds_key_map.get(best_outcome)
        pred_odds = row.get(pred_odds_key)
        
        if pred_odds and pred_odds > 1:
            stake_pool += 1
            if best_outcome == actual:
                roi_pool += (pred_odds - 1)
            else:
                roi_pool -= 1
    
    acc = correct / total if total > 0 else 0
    ll = log_loss / total if total > 0 else 0
    roi = roi_pool / stake_pool if stake_pool > 0 else 0
    
    logger.info("="*40)
    logger.info(f"TEST RESULTS (N={total})")
    logger.info("="*40)
    logger.info(f"Accuracy: {acc:.2%}")
    logger.info(f"Log Loss: {ll:.4f}")
    logger.info(f"ROI (Flat Stake on Fav): {roi:.2%}")
    logger.info("="*40)
    
    if acc > 0.45:
        logger.info("✅ Accuracy Acceptable (>45%)")
    else:
        logger.warning("⚠️ Accuracy Low (<45%)")
        
    if roi > -0.05:
        logger.info("✅ ROI Acceptable (>-5% flat staking is hard)")
    else:
        logger.warning(f"⚠️ ROI Poor ({roi:.2%})")

if __name__ == "__main__":
    main()
