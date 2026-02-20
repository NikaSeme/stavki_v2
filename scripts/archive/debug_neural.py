
import logging
import sys
import pandas as pd
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from stavki.models.neural.multitask import MultiTaskModel

def main():
    logger.info("DEBUG: Starting Neural Test")
    
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    if not data_path.exists():
        logger.error("Data not found")
        return

    logger.info("DEBUG: Loading data...")
    df = pd.read_csv(data_path)
    logger.info(f"DEBUG: Loaded {len(df)} rows. Taking top 500 for test.")
    
    df = df.head(500)
    
    logger.info("DEBUG: Initializing Model...")
    nn = MultiTaskModel(n_epochs=1, batch_size=32)
    
    logger.info("DEBUG: calling fit()...")
    try:
        nn.fit(df, eval_ratio=0.2)
        logger.info("DEBUG: fit() completed successfully!")
    except Exception as e:
        logger.exception("DEBUG: fit() failed")

if __name__ == "__main__":
    main()
