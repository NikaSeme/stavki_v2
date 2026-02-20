
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.data.datasets import StavkiDataset

def test_dataset():
    print("🧪 Testing StavkiDataset...")
    
    try:
        ds = StavkiDataset()
        print(f"✅ Loaded {len(ds)} matches.")
        print(f"   Num Players: {ds.num_players}")
        
        if len(ds) == 0:
            print("⚠️ Dataset is empty. Check parquets.")
            return

        item = ds[0]
        print("\nItem 0 Shapes:")
        for k, v in item.items():
            if torch.is_tensor(v):
                print(f"   {k}: {v.shape} ({v.dtype})")
            else:
                print(f"   {k}: {v}")
                
        # Test Batch
        dl = DataLoader(ds, batch_size=4, shuffle=True)
        batch = next(iter(dl))
        print("\nBatch Shapes (B=4):")
        print(f"   Home Context: {batch['home_context'].shape}")
        print(f"   Sample Context (First Team): {batch['home_context'][0]}")
        print(f"   Outcome: {batch['outcome'].shape}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset()
