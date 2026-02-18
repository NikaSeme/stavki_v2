
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import gc
import pickle
import torch
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from stavki.models.base import Market, BaseModel
from stavki.strategy.optimizer import WeightOptimizer

def main():
    logger.info("🔧 Starting Debug Optimization...")
    
    # 1. Load Data
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    
    # 2. Split Data (Temporal)
    # 60% Train, 20% Val (Early Stopping), 20% Test (Calibration)
    train_ratio = 0.60
    val_ratio = 0.20
    
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    
    # Validation targets
    if "target" not in val_df.columns:
        if "FTR" in val_df.columns:
             val_df["target"] = val_df["FTR"]
    
    target_map = {"H": "home", "D": "draw", "A": "away"}
    if "target" in val_df.columns:
        actuals = val_df["target"].map(target_map).dropna()
    else:
        logger.error("No targets found in val_df")
        sys.exit(1)

    logger.info(f"Validation Set: {len(val_df)} matches")

    # 3. Load Models
    models_dir = PROJECT_ROOT / "models"
    model_names = ["dixon_coles.pkl", "LightGBM_1X2.pkl", "catboost.pkl", "neural_multitask.pkl"]
    loaded_models = {}
    
    # Map filenames to keys used in optimizer
    key_map = {
        "dixon_coles.pkl": "poisson",
        "LightGBM_1X2.pkl": "lightgbm",
        "catboost.pkl": "catboost",
        "neural_multitask.pkl": "neural"
    }

    for fname in model_names:
        path = models_dir / fname
        key = key_map.get(fname)
        if path.exists():
            logger.info(f"Loading {fname}...")
            try:
                # Use robust loading logic (copied from base.py mostly or just standard load)
                # But we can import the classes and use .load()
                # We need to identify class?
                # BaseModel.load() handles it if we import subclasses?
                # Let's import them.
                from stavki.models.poisson.dixon_coles import DixonColesModel
                from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
                from stavki.models.catboost.catboost_model import CatBoostModel
                from stavki.models.neural.multitask import MultiTaskModel
                
                # We use the specific class .load if possible, or BaseModel.load
                if "dixon" in fname:
                    m = DixonColesModel.load(path)
                elif "LightGBM" in fname:
                    m = LightGBMModel.load(path)
                elif "catboost" in fname:
                    m = CatBoostModel.load(path)
                elif "neural" in fname:
                    m = MultiTaskModel.load(path)
                
                loaded_models[key] = m
                logger.info(f"Loaded {key}")
            except Exception as e:
                logger.error(f"Failed to load {fname}: {e}")
        else:
            logger.warning(f"Model file missing: {path}")

    # 4. Generate Predictions
    logger.info("Generating predictions...")
    model_preds = {}
    
    for key, model in loaded_models.items():
        logger.info(f"Predicting with {key}...")
        try:
            preds = model.predict(val_df)
            # Convert to DF
            filtered = [p for p in preds if p.market == Market.MATCH_WINNER]
            df_preds = pd.DataFrame([
                {"home": p.probabilities.get("home", 0), "draw": p.probabilities.get("draw", 0), "away": p.probabilities.get("away", 0)}
                for p in filtered
            ])
            model_preds[key] = df_preds
        except Exception as e:
            logger.error(f"Prediction failed for {key}: {e}")

    # 5. Optimize
    optimizer = WeightOptimizer()
    
    if "League" not in val_df.columns:
        val_df["League"] = "Unknown"
        
    leagues = val_df["League"].unique().tolist()
    logger.info(f"Leagues to optimize: {leagues}")
    
    results = {}
    
    # Manual loop to debug and save incrementally
    config_path = PROJECT_ROOT / "stavki" / "config" / "leagues.json"
    
    # Load existing config to preserve
    league_config = {}
    if config_path.exists():
        with open(config_path) as f:
            try:
                league_config = json.load(f)
            except:
                pass

    for league in leagues:
        logger.info(f"Creating mask for {league}...")
        mask = val_df["League"] == league
        league_actuals = actuals[actuals.index.isin(val_df[mask].index)]
        
        if len(league_actuals) < 10:
            logger.warning(f"Skipping {league}: too few samples ({len(league_actuals)})")
            continue
            
        logger.info(f"Optimizing {league} ({len(league_actuals)} samples)...")
        
        # Prepare league preds
        league_preds = {k: v[mask].reset_index(drop=True) for k, v in model_preds.items() if not v.empty}
        
        # Optimize single league
        try:
            # We use the internal _optimize_single_league if available, or just call optimize_per_league with 1 league
            # But optimize_per_league takes all preds.
            # Let's use optimize_per_league but pass filtered df?
            # No, optimize_per_league expects full aligned dfs.
            # Let's just use the optimizer's logic here directly or call it per league?
            # WeightOptimizer.optimize_per_league iterates.
            # We can just call it with current `val_df[mask]`?
            # But actuals need to align.
            
            # Let's inspect WeightOptimizer in a bit. For now, try calling it on restricted set?
            # Models expect full DF? No, they already predicted.
            # model_predictions is a dict of DFs.
            # If we slice them, they are new DFs.
            
            # Slice everything
            sliced_preds = {k: v[mask].reset_index(drop=True) for k, v in model_preds.items()}
            sliced_actuals = league_actuals.reset_index(drop=True)
            sliced_odds = val_df[mask].reset_index(drop=True)
            
            # Calling optimize_per_league on specific league list (size 1)
            # It will iterate that one league.
            # But we pass sliced data where "League" column is all `league`.
            
            logger.info(f"Calling optimizer for {league}...")
            # We need to make sure 'League' column exists in sliced_odds for the optimizer to group by it.
            # It does.
            
            league_res = optimizer.optimize_per_league(
                model_predictions=sliced_preds,
                actual_outcomes=sliced_actuals,
                odds_data=sliced_odds,
                leagues=[league],
                metric="log_loss"
            )
            
            weights = league_res.get(league)
            if weights:
                results[league] = weights
                logger.info(f"✅ Optimized {league}: {weights}")
                
                # Save immediately
                if league not in league_config:
                    league_config[league] = {}
                league_config[league]["weights"] = weights
                
                with open(config_path, "w") as f:
                    json.dump(league_config, f, indent=2)
            else:
                logger.warning(f"No weights returned for {league}")

        except Exception as e:
            logger.error(f"Optimization failed for {league}: {e}")
            import traceback
            traceback.print_exc()

    logger.info("Debug Optimization Complete!")

if __name__ == "__main__":
    main()
