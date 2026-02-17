
import pandas as pd
from pathlib import Path
from stavki.config import DATA_DIR, PROJECT_ROOT

def main():
    path = DATA_DIR / "features_full.parquet"
    if not path.exists():
        path = PROJECT_ROOT / "data" / "features_full.parquet"
        
    if not path.exists():
        print(f"File not found at {path}")
        return
    
    df = pd.read_parquet(path)
    print(f"Total Columns: {len(df.columns)}")
    print("Columns:")
    for c in sorted(df.columns):
        print(f"{c} type={df[c].dtype}")

if __name__ == "__main__":
    main()
