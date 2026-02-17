
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from stavki.config import PROJECT_ROOT, DATA_DIR
from stavki.models.training.trainer import ModelTrainer
from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
from stavki.models.ensemble import EnsemblePredictor, EnsembleCalibrator
from stavki.models.base import Market

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting LightGBM Retraining (Leakage Fix)...")
    
    # 1. Load Data
    data_path = DATA_DIR / "features_full.parquet"
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Sort by date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date")
    
    # 2. Load Existing Models
    models_dir = PROJECT_ROOT / "models"
    trainer = ModelTrainer(models_dir=models_dir)
    trainer.load_models() # Loads Neural, Poisson, etc.
    
    # 3. Re-initialize LightGBM_1X2
    # This ensures we use the NEW code with exclusion list logic
    logger.info("Re-initializing LightGBM_1X2...")
    lgbm = LightGBMModel()
    trainer.models["LightGBM_1X2"] = lgbm # Replace old model
    
    # 4. Prepare Splits (Logic from Trainer)
    n = len(df)
    train_end = int(n * trainer.train_ratio)
    cal_end = int(n * (trainer.train_ratio + trainer.cal_ratio))
    
    train_df = df.iloc[:train_end]
    cal_df = df.iloc[train_end:cal_end]
    
    # 5. Train LightGBM
    logger.info("Training LightGBM_1X2...")
    start_time = datetime.now()
    metrics = lgbm.fit(train_df)
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"LightGBM_1X2 trained in {elapsed:.1f}s")
    logger.info(f"Metrics: {metrics}")
    
    # Save immediately
    lgbm.save(models_dir / "LightGBM_1X2.pkl")
    
    # 6. Re-build Ensemble
    logger.info("Re-building Ensemble...")
    trainer.ensemble = EnsemblePredictor(models=trainer.models)
    
    # 7. Fit Calibration (Need predictions from all models on cal set)
    logger.info("Generating calibration predictions...")
    cal_predictions = []
    for name, model in trainer.models.items():
        if model.is_fitted:
            try:
                preds = model.predict(cal_df)
                cal_predictions.extend(preds)
            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")
    
    logger.info("Fitting Calibrator...")
    trainer.calibrator = EnsembleCalibrator(method="isotonic")
    
    # Build actuals for calibrator
    actuals = {}
    for idx, row in cal_df.iterrows():
        match_id = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}_{idx}")
        # 1X2 actual
        if row["FTHG"] > row["FTAG"]:
            actuals[match_id] = "home"
        elif row["FTHG"] < row["FTAG"]:
            actuals[match_id] = "away"
        else:
            actuals[match_id] = "draw"
            
    trainer.calibrator.fit(cal_predictions, actuals)
    
    # 8. Optimize Ensemble Weights
    logger.info("Optimizing Ensemble Weights...")
    markets = [Market.MATCH_WINNER, Market.OVER_UNDER, Market.BTTS]
            
    for market in markets:
        try:
            weights = trainer.ensemble.optimize_weights(cal_df, market)
            logger.info(f"New weights for {market.value}: {weights}")
        except Exception as e:
            logger.warning(f"Weight optimization failed for {market.value}: {e}")
            
    # 9. Save All
    logger.info("Saving updated models and ensemble...")
    trainer._save_all_models()
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
