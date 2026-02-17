
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

from stavki.config import PROJECT_ROOT, DATA_DIR
from stavki.models.training.trainer import ModelTrainer
from stavki.models.catboost.catboost_model import CatBoostModel
from stavki.models.neural.multitask import MultiTaskModel
from stavki.models.gradient_boost import LightGBMModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Full-Scale Retraining...")
    
    # 1. Load Data
    data_path = DATA_DIR / "features_full.parquet"
    if not data_path.exists():
        logger.warning(f"Parquet file not found at {data_path}, trying CSV...")
        data_path = DATA_DIR / "features_full.csv"
    
    if not data_path.exists():
        logger.error("No training data found! Please run feature generation first.")
        return
        
    logger.info(f"Loading data from {data_path}...")
    if str(data_path).endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)
        
    logger.info(f"Loaded {len(df)} rows.")
    
    # 2. Initialize Trainer (70% train, 15% cal, 15% test)
    models_dir = PROJECT_ROOT / "models"
    trainer = ModelTrainer(models_dir=models_dir)
    
    # 3. Add Models
    # Default adds: DixonColes, LightGBM, NeuralMultiTask, GoalsRegressor
    trainer.add_default_models()
    
    # Explicitly add CatBoost (part of new architecture)
    # 1000 iterations, use GPU if available (but forcing false for stability)
    catboost = CatBoostModel(iterations=1000, use_gpu=False) 
    trainer.add_model(catboost)


    
    # 4. Train All
    logger.info("Training models and optimizing ensemble...")
    # Get list of leagues for per-league optimization
    leagues = df["League"].unique().tolist() if "League" in df.columns else None
    
    results = trainer.train_all(
        data=df,
        optimize_weights=True, # Uses new SLSQP optimizer
        leagues=leagues
    )
    
    # 5. Print Summary
    logger.info("\n=== Training Summary ===")
    for model, info in results["models"].items():
        status = info.get("status", "unknown")
        metrics = info.get("metrics", {})
        if status == "success":
            acc = metrics.get('accuracy_1x2') or metrics.get('eval_accuracy')
            brier = metrics.get('brier_score') or metrics.get('eval_log_loss')
            logger.info(f"{model:<20} | Status: {status} | Acc: {acc:.4f} | Loss: {brier:.4f}")
        else:
            logger.info(f"{model:<20} | Status: {status} | Error: {info.get('error')}")
            
    if "weights_1x2" in results:
        logger.info(f"\nEnsemble Weights (1X2): {results['weights_1x2']}")
        
    logger.info(f"\nModels saved to {models_dir}")
    logger.info("Full-Scale Retraining Complete.")

if __name__ == "__main__":
    main()
