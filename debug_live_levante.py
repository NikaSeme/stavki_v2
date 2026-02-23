import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/macuser/Documents/something/stavki_v2")
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from stavki.data.schemas.match import Match, Team
from datetime import datetime

# Mock the match
m = Match(
    id="test_barca_levante",
    league="soccer_spain_la_liga",
    season="2023/2024",
    commence_time=datetime.now(),
    home_team=Team(id="t1", name="FC Barcelona", normalized_name="fc barcelona"),
    away_team=Team(id="t2", name="Levante", normalized_name="levante")
)

df = pd.DataFrame([m])
# We need to run _build_features similar to daily pipeline
from stavki.pipelines.daily import DailyPipeline
pipeline = DailyPipeline()
pipeline._init_components()

# Enrich the match? Just try to build basic features first to see what TeamMapper outputs
# Let's see what the mapped names are.
from stavki.data.processors.normalize import TeamMapper
mapper = TeamMapper()
print(f"Barca mapped: {mapper.map_name('FC Barcelona')}")
print(f"Levante mapped: {mapper.map_name('Levante')}")
print(f"Roma mapped: {mapper.map_name('Roma')}")
print(f"Cremonese mapped: {mapper.map_name('Cremonese')}")
print(f"Parma mapped: {mapper.map_name('Parma')}")

# Extract features directly from Redis or the Registry
from stavki.features.registry import FeatureRegistry
registry = FeatureRegistry()
# Mock the dataframe with the bare minimum to invoke registry
match_df = pd.DataFrame([{
    'match_id': 'test_barca_levante',
    'date': datetime.now(),
    'league': 'LA_LIGA',
    'home_team': mapper.map_name('FC Barcelona') or 'fc barcelona',
    'away_team': mapper.map_name('Levante') or 'levante',
    'home_manager': 'Xavi',
    'away_manager': 'Unknown',
    'referee': 'Unknown'
}])

try:
    X_live = registry.build_live_features(match_df)
    print("\n--- Features for Barca vs Levante ---")
    for col in X_live.columns:
        if 'elo' in col.lower() or 'xg' in col.lower() or 'rating' in col.lower() or 'form' in col.lower():
            print(f"{col}: {X_live[col].values[0]}")
except Exception as e:
    print(f"Registry failed: {e}")
