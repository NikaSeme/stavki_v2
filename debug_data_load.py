
import pandas as pd
from pathlib import Path
import sys

def test_load():
    paths = [
        Path("data/training_data.csv"),
        Path("data/features_full.csv"),
    ]
    
    for p in paths:
        print(f"Checking {p}...")
        if not p.exists():
            print(f"  - Does not exist")
            continue
            
        try:
            df = pd.read_csv(p, low_memory=False)
            print(f"  - SUCCESS: Loaded {len(df)} rows")
        except Exception as e:
            print(f"  - FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_load()
