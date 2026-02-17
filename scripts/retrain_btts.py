
import logging
import pandas as pd
from pathlib import Path
import pickle

from stavki.config import PROJECT_ROOT, DATA_DIR
from stavki.models.gradient_boost.btts_model import BTTSModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Targeted BTTS Retraining...")
    
    # 1. Load Data
    data_path = DATA_DIR / "features_full.csv"
    
    if not data_path.exists():
        logger.error(f"No training data found at {data_path}")
        return
        
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    logger.info(f"Loaded {len(df)} rows.")
    
    # 2. Initialize Model
    logger.info("Initializing BTTSModel (with snake_case features)...")
    model = BTTSModel(n_estimators=500)
    
    # 3. Validation Split
    # Manual split to match system training
    # 70% train, 15% eval, 15% test
    n = len(df)
    train_end = int(n * 0.70)
    
    # 4. Train
    logger.info("Training...")
    metrics = model.fit(df) # fit handles its own split if eval_ratio provided, default 0.2
    # But usually we want to respect time split.
    # The fit method in LightGBMModel sorts by date and uses last 20% (eval_ratio=0.2).
    # This is fine for targeted retraining.
    
    logger.info(f"Training Complete. Metrics: {metrics}")
    
    # 5. Save
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    save_path = models_dir / "LightGBM_BTTS.pkl"
    
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")
    
    # 6. Verify Features
    logger.info(f"Model expects features: {model.metadata.get('features')}")
    
if __name__ == "__main__":
    main()
