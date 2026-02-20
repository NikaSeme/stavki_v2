
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

import random
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # torch will be imported inside implementation or we can import it here if safe
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Imports
from stavki.models.poisson.dixon_coles import DixonColesModel
from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
from stavki.models.gradient_boost.btts_model import BTTSModel
from stavki.models.catboost.catboost_model import CatBoostModel
from stavki.models.neural.multitask import MultiTaskModel
from stavki.models.neural.goals_regressor import GoalsRegressor
from stavki.strategy.optimizer import WeightOptimizer
from stavki.models.ensemble.predictor import EnsemblePredictor

def main():
    set_seed(42) # Ensure reproducibility
    logger.info("🚀 Starting Full System Retraining...")
    
    # 1. Load Data
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    if df.empty:
        logger.error("Data file is empty!")
        sys.exit(1)
        
    logger.info(f"Loaded {len(df)} matches. Sorting by date...")

    # Compatibility: Alias snake_case columns to CamelCase if missing
    # NeuralMultiTask and DixonColes expect CamelCase (HomeTeam, League)
    aliases = {
        "home_team": "HomeTeam",
        "away_team": "AwayTeam", 
        "league": "League",
        "date": "Date",
        "home_score": "FTHG",
        "away_score": "FTAG"
    }
    for search_col, target_col in aliases.items():
        if target_col not in df.columns and search_col in df.columns:
            logger.info(f"Aliasing column: {search_col} -> {target_col}")
            df[target_col] = df[search_col]

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # CRITICAL: Normalize team names to match Live Pipeline
    # This ensures models are trained on the exact same canonical names used in production
    logger.info("Normalizing team names...")
    from stavki.data.processors.normalize import normalize_team_name
    if 'HomeTeam' in df.columns:
        df['HomeTeam'] = df['HomeTeam'].apply(normalize_team_name)
    if 'AwayTeam' in df.columns:
        df['AwayTeam'] = df['AwayTeam'].apply(normalize_team_name)
    
    # OPTIMIZATION: Downcast float64 to float32 to save 50% RAM
    fcols = df.select_dtypes('float').columns
    df[fcols] = df[fcols].astype(np.float32)
    gc.collect()
    
    # 2. Split Data (Temporal)
    # 60% Train, 20% Val (Early Stopping), 20% Test (Calibration)
    train_ratio = 0.60
    val_ratio = 0.20
    # test_ratio implicit = 1.0 - 0.8 = 0.20
    
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    
    # Slice using boundaries for clarity
    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)
    
    # OPTIMIZATION: Free original dataframe immediately
    del df
    gc.collect()
    
    logger.info(f"Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # 3. Train Models
    models = {}
    
    # Create models directory
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    # Poisson
    logger.info("\n📉 Training Poisson (Dixon-Coles)...")
    dc = DixonColesModel()
    dc.fit(train_df) 
    dc.save(models_dir / "dixon_coles.pkl")
    models["poisson"] = dc
    
    # LightGBM 1X2
    logger.info("\n⚡ Training LightGBM (1X2)...")
    lgb = LightGBMModel(n_estimators=500, learning_rate=0.05)
    lgb.fit(train_df, eval_ratio=0.2) 
    lgb.save(models_dir / "LightGBM_1X2.pkl")
    models["lightgbm"] = lgb

    # LightGBM BTTS
    logger.info("\n⚡ Training LightGBM (BTTS)...")
    btts = BTTSModel(n_estimators=300, learning_rate=0.05)
    btts.fit(train_df, eval_ratio=0.2)
    btts.save(models_dir / "LightGBM_BTTS.pkl")
    models["btts"] = btts
    
    # CatBoost (Leakage Fixed)
    logger.info("\n🐱 Training CatBoost...")
    # CatBoostModel likely takes no args or different args in init
    # We will set params via internal defaults or kwargs if supported
    cb = CatBoostModel() 
    # Use kwargs in fit if needed, or rely on internal defaults
    # based on previous file view, it has hardcoded params or uses kwargs in fit?
    # Let's check the file view result first.
    # But for now I'll assume empty init based on error.
    cb.fit(train_df, eval_ratio=0.2)
    cb.save(models_dir / "catboost.pkl")
    models["catboost"] = cb
    
    # Neural MultiTask (Feature Fixed)
    logger.info("\n🧠 Training Neural MultiTask...")
    # OPTIMIZATION: Gradient Accumulation (16 * 4 = 64 Effective Batch)
    nn = MultiTaskModel(n_epochs=25, batch_size=16) 
    nn.fit(train_df, eval_ratio=0.2, accumulation_steps=4, num_workers=0, pin_memory=False)
    nn.save(models_dir / "neural_multitask.pkl")
    models["neural"] = nn
    
    # Goals Regressor (Feature Fixed)
    logger.info("\n⚽ Training Goals Regressor...")
    gr = GoalsRegressor(n_epochs=30, batch_size=16) 
    gr.fit(train_df, accumulation_steps=4, num_workers=0, pin_memory=False)
    gr.save(models_dir / "goals_regressor.pkl")
    models["goals"] = gr
    
    # 4. Optimize Ensemble
    logger.info("\n⚖️ Optimizing Ensemble Weights...")
    
    # Generate predictions on Validation set
    logger.info("Generating validation predictions...")
    preds_dc = dc.predict(val_df)
    preds_lgb = lgb.predict(val_df)
    preds_cb = cb.predict(val_df)
    preds_nn = nn.predict(val_df)
    
    optimizer = WeightOptimizer()
    
    from stavki.models.base import Market
    
    def preds_to_df(preds):
        filtered = [p for p in preds if p.market == Market.MATCH_WINNER]
        return pd.DataFrame([
            {"home": p.probabilities.get("home", 0), "draw": p.probabilities.get("draw", 0), "away": p.probabilities.get("away", 0)}
            for p in filtered
        ])
    
    model_preds = {
        "poisson": preds_to_df(preds_dc),
        "lightgbm": preds_to_df(preds_lgb),
        "catboost": preds_to_df(preds_cb),
        "neural": preds_to_df(preds_nn),
    }
    
    # Prepare targets
    if "target" not in val_df.columns:
        if "FTR" in val_df.columns:
             val_df["target"] = val_df["FTR"]
        else:
             logger.warning("No FTR/target found in validation set. Optimization might fail.")
    
    # Map H/D/A to home/draw/away
    target_map = {"H": "home", "D": "draw", "A": "away"}
    if "target" in val_df.columns:
        actuals = val_df["target"].map(target_map)
        actuals = actuals.dropna()
    else:
        logger.error("Could not determine actual outcomes for optimization.")
        actuals = pd.Series()

    if "League" not in val_df.columns:
        val_df["League"] = "Unknown"
        
    leagues = val_df["League"].unique().tolist()
    
    # Optimize
    if not actuals.empty:
        results = optimizer.optimize_per_league(
            model_predictions=model_preds,
            actual_outcomes=actuals,
            odds_data=val_df,
            leagues=leagues,
            metric="log_loss" 
        )
        
        # Save optimized weights
        config_path = PROJECT_ROOT / "stavki" / "config" / "leagues.json"
        
        league_config = {}
        if config_path.exists():
            import json
            with open(config_path) as f:
                try:
                    league_config = json.load(f)
                except:
                    pass
                
        # Update weights
        for league, weights in results.items():
            if league not in league_config:
                league_config[league] = {}
            league_config[league]["weights"] = weights
            logger.info(f"Updated {league}: {weights}")
            
        with open(config_path, "w") as f:
            import json
            json.dump(league_config, f, indent=2)
            
        logger.info(f"Saved optimized weights to {config_path}")
    else:
        logger.warning("Skipping optimization due to missing targets.")
    
    logger.info("\n✅ Retraining Complete!")

if __name__ == "__main__":
    main()
