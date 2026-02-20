
import pandas as pd
from pathlib import Path

csv_path = Path("data/features_full.csv")
if not csv_path.exists():
    print("❌ data/features_full.csv not found")
    exit(1)

df = pd.read_csv(csv_path, low_memory=False)
print(f"Loaded {len(df)} rows")

required_cols = ["xg_home", "xg_away", "xg_diff", "xg_efficiency_home", "xg_efficiency_away"]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    print(f"❌ Missing columns: {missing}")
    exit(1)

print("✅ All xG columns present.")

# Check for non-nulls
for col in required_cols:
    non_null = df[col].notna().sum()
    print(f"{col}: {non_null}/{len(df)} non-null entries")
    if non_null == 0:
        print(f"⚠️  {col} is completely empty!")

# Check values (should not be all defaults)
defaults = {
    "xg_home": 1.35,
    "xg_away": 1.15
}

for col, default_val in defaults.items():
    if col in df.columns:
        mean_val = df[col].mean()
        num_defaults = (df[col] == default_val).sum()
        print(f"{col} mean: {mean_val:.3f}. Entries = default ({default_val}): {num_defaults}/{len(df)}")
