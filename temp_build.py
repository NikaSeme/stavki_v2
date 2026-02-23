import pandas as pd
from pathlib import Path
from stavki.pipelines.daily import DailyPipeline
from stavki.features.registry import FeatureRegistry

DATA_DIR = Path("data")

print("Loading DataFrames...")
# We need to merge enriched (lineups, xG) with full (scores, simple stats)
df_enriched = pd.read_parquet(DATA_DIR / "features_enriched.parquet")
df_full = pd.read_csv(DATA_DIR / "features_full.csv", low_memory=False)

# Get only the essential result/stat columns from full CSV
stat_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
stat_cols = [c for c in stat_cols if c in df_full.columns]

print(f"Merging {len(df_enriched)} enriched matches with {len(df_full)} full matches...")
df_full_idx = df_full[stat_cols].reset_index(names='csv_index')
df_merged = df_enriched.merge(df_full_idx, on='csv_index', how='left')

print("Parsing DataFrames into Pydantic Match objects...")
pipe = DailyPipeline()
matches = pipe._df_to_matches(df_merged, is_historical=True)

completed = [m for m in matches if m.is_completed]
print(f"Total Matches: {len(matches)} | Completed: {len(completed)}")

print("Running FeatureRegistry Transform...")
registry = FeatureRegistry(training_mode=False)
df_features = registry.transform_historical(matches)

print("\nSuccess! Generated Features Layout:")
print(df_features.head())
print(f"Total computed features length: {len(df_features)}")
