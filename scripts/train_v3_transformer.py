
import sys
from pathlib import Path
import logging
import pandas as pd
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_v3")

from stavki.models.neural.transformer_model import TransformerModel

def main():
    logger.info("Initializing V3 Transformer Training...")
    
    # 1. Load Data
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
        
    df = pd.read_csv(data_path, low_memory=False)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
    
    logger.info(f"Loaded {len(df)} matches.")
    
    # 2. Instantiate Model
    model = TransformerModel(
        seq_len=10,
        d_model=64,
        n_heads=4,
        n_layers=2
    )
    
    # 3. Train
    # We use a small number of epochs for the "deployment" test, 
    # but enough to get weights.
    # The user can run this with more epochs later.
    epochs = 5
    logger.info(f"Training for {epochs} epochs...")
    
    metrics = model.fit(df, epochs=epochs, batch_size=32, lr=0.001)
    logger.info(f"Training complete. Metrics: {metrics}")
    
    # 4. Save
    save_path = PROJECT_ROOT / "models" / "v3_transformer.pth"
    torch.save(model._get_state(), save_path)
    logger.info(f"Saved V3 model to {save_path}")

if __name__ == "__main__":
    main()
