
import pandas as pd
from stavki.features.registry import FeatureRegistry
from stavki.data.schemas import Match, Team, League
from datetime import datetime

def check():
    # 1. Load CSV headers
    print("Loading CSV headers...")
    try:
        df = pd.read_csv("data/features_full.csv", nrows=0)
        csv_cols = set(df.columns)
        print(f"CSV has {len(csv_cols)} columns")
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # 2. Get Registry features
    print("Generating Registry features...")
    registry = FeatureRegistry()
    registry._is_fitted = True  # Hack to bypass check
    registry._historical_matches = [] 
    
    # Mock data to generic keys
    keys = registry.get_feature_names()
    reg_cols = set(keys)
    print(f"Registry produces {len(reg_cols)} features")

    # 3. Compare
    print("\nMissing in Registry (Present in CSV):")
    missing = csv_cols - reg_cols
    # Filter out non-feature cols
    ignore = {"HomeTeam", "AwayTeam", "Date", "FTR", "Result", "Season", "League", "Div", "Time", "Ref", "Referee"}
    missing = {c for c in missing if c not in ignore and not c.startswith("B365") and not c.startswith("Bb")}
    
    for c in sorted(missing):
        print(f" - {c}")

    print("\nExtra in Registry (Missing in CSV):")
    extra = reg_cols - csv_cols
    for c in sorted(extra):
        print(f" + {c}")

if __name__ == "__main__":
    check()
