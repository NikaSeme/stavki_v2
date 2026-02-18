
import sys
import pandas as pd
from pathlib import Path
import logging

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from stavki.models.neural.transformer_model import TransformerModel

def main():
    print("🤖 Training V3 Transformer...")
    
    # Load Data
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    if not data_path.exists():
        print("Data not found!")
        return
        
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Instantiate
    model = TransformerModel(seq_len=10, d_model=32, n_heads=4, n_layers=2)
    
    # Fit
    # This expects df to have FTHG/FTAG/Elo etc.
    metrics = model.fit(df, epochs=5, batch_size=32)
    
    print("\n✅ Training Complete")
    print(f"Final Val Loss: {metrics['val_loss']:.4f}")
    print(f"Final Val Acc: {metrics['val_acc']:.4%}")
    
    # Compare with Baseline
    # Log loss < 0.994 is the target (Market Baseline)
    if metrics['val_loss'] < 0.994:
        print("\n🏆 BEATS MARKET BASELINE!")
    else:
        print("\n⚠️ Does not beat market baseline yet.")

if __name__ == "__main__":
    main()
