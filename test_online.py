import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path("/Users/macuser/Documents/something/stavki_v2")
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.pipelines.daily import DailyPipeline
from stavki.features.registry import FeatureRegistry

print("Loading silver matches...")
df = pd.read_parquet(PROJECT_ROOT / "data/processed/matches/matches_silver.parquet")
df['date'] = pd.to_datetime(df['date'])

# Filter for 2026 matches
recent_df = df[df['date'] > '2026-01-01'].copy()
print(f"Found {len(recent_df)} recent matches")

# Map columns to what DailyPipeline might expect, or just see if the system works with silver schema?
# Actually, process_matches.py already mapped the JSONs to silver parity.

# Let's inspect what DailyPipeline._df_to_matches() expects
import inspect
print(inspect.getsource(DailyPipeline._df_to_matches))
