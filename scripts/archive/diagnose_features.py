#!/usr/bin/env python3
"""Diagnostic: show what features each model actually sees at prediction time"""
import sys, logging
from pathlib import Path
import pickle
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
for n in ["httpx","urllib3","httpcore"]: logging.getLogger(n).setLevel(logging.WARNING)

def main():
    print("=== PIPELINE FEATURE CAPTURE ===")
    from stavki.pipelines.daily import DailyPipeline
    pipeline = DailyPipeline()
    
    print("1. Fetching odds...")
    odds_df = pipeline._fetch_odds()
    
    print("2. Extracting matches...")
    matches_df = pipeline._extract_matches(odds_df)
    print(f"   Found {len(matches_df)} matches")
    
    print("3. Enriching matches (SportMonks)...")
    pipeline._enrich_matches(matches_df)
    
    print("4. Building features...")
    features_df = pipeline._build_features(matches_df, odds_df)
    print(f"   Built features DF: {features_df.shape}")

    print("\n=== CATBOOST MODEL INSPECTION ===")
    with open("models/catboost.pkl", "rb") as f:
        state = pickle.load(f)
    
    # Handle dict vs object
    if isinstance(state, dict):
        metadata = state.get("metadata", {})
        model_features = metadata.get("features", [])
        cat_features = metadata.get("cat_features", [])
    else:
        model_features = getattr(state, "features", [])
        cat_features = getattr(state, "cat_features", [])
        
    print(f"Model expects {len(model_features)} features.")
    if model_features:
        print(f"First 10: {model_features[:10]}")

    if not features_df.empty:
        print("\n=== DATA QUALITY AUDIT ===")
        
        # 1. Check ELO defaults (1500.0)
        elo_cols = [c for c in features_df.columns if "elo" in c]
        for c in elo_cols:
            defaults = (features_df[c] == 1500.0).sum()
            pct = (defaults / len(features_df)) * 100
            print(f"{c:20s}: {defaults:>2}/{len(features_df)} ({pct:.1f}%) are 1500.0 (default)")
            
        # 2. Check Form defaults (0)
        form_cols = [c for c in features_df.columns if "form" in c and "pts" in c]
        for c in form_cols:
            zeros = (features_df[c] == 0).sum()
            pct = (zeros / len(features_df)) * 100
            print(f"{c:20s}: {zeros:>2}/{len(features_df)} ({pct:.1f}%) are 0 (potential default)")
            
            # Show which teams have 0 form
            if zeros > 0:
                zero_rows = features_df[features_df[c] == 0]
                teams = zero_rows["home_team" if "home" in c else "away_team"].unique()
                print(f"    Teams with 0 {c}: {list(teams)[:5]}...")

        # 3. Check Implied Probs defaults (0.5)
        imp_cols = [c for c in features_df.columns if "imp" in c]
        for c in imp_cols:
            defaults = (features_df[c] == 0.5).sum()
            pct = (defaults / len(features_df)) * 100
            print(f"{c:20s}: {defaults:>2}/{len(features_df)} ({pct:.1f}%) are 0.5 (default)")

        # 4. Check Missing Features
        if model_features:
            missing = [f for f in model_features if f not in features_df.columns]
            if missing:
                print(f"\nCRITICAL: {len(missing)} features missing from live dataframe!")
                print(f"Example missing: {missing[:10]}")
            else:
                print("\nAll model features present in live dataframe.")

        # 5. Show First Match Details
        print("\n=== SAMPLE MATCH FEATURES ===")
        row = features_df.iloc[0]
        print(f"Match: {row.get('home_team')} vs {row.get('away_team')}")
        print(f"League: {row.get('league', 'unknown')}")
        
        cols_to_show = elo_cols + form_cols + imp_cols + ["AvgH", "AvgD", "AvgA"]
        for c in cols_to_show:
            if c in row.index:
                print(f"{c:20s}: {row[c]}")

if __name__ == "__main__":
    main()
