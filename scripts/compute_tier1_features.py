#!/usr/bin/env python3
"""
Compute Tier 1 Features (REFACTORED for Unified Live Parity)
========================

Uses FeatureRegistry to chronologically evaluate features_enriched.parquet.
This ensures mathematically identical feature logic offline vs. live scoring.

Usage:
    python3 scripts/compute_tier1_features.py --merge
"""

import sys
import logging
import argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.pipelines.daily import DailyPipeline
from stavki.features.registry import FeatureRegistry

DATA_DIR = PROJECT_ROOT / "data"
ENRICHED_PATH = DATA_DIR / "features_enriched.parquet"
FEATURES_CSV = DATA_DIR / "features_full.csv"
OUTPUT_PATH = DATA_DIR / "features_tier1.parquet"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("tier1_unified")


def merge_into_csv(df_features: pd.DataFrame, df_enriched: pd.DataFrame, csv_path: Path):
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return
        
    logger.info(f"Loading {csv_path} for merging...")
    df_csv = pd.read_csv(csv_path, low_memory=False)
    original_cols = len(df_csv.columns)
    
    # We need a csv_index bridging logic. The df_features dataframe maintains match_id.
    # match_id is the fixture_id. We need to map fixture_id -> csv_index using df_enriched.
    mapping_df = df_enriched[['fixture_id', 'csv_index']].dropna().drop_duplicates(subset=['fixture_id'])
    # Convert fixture_id natively and safely drop alphanumeric corrupted indexes
    mapping_df['fixture_id'] = pd.to_numeric(mapping_df['fixture_id'], errors='coerce')
    mapping_df = mapping_df.dropna(subset=['fixture_id'])
    mapping_df['fixture_id'] = mapping_df['fixture_id'].astype(int)
    
    # Ensure df_features has int match_id
    if 'match_id' in df_features.columns:
        df_features['match_id'] = df_features['match_id'].astype(int)
    
    # Attach csv_index to df_features
    df_feat_mapped = df_features.merge(mapping_df, left_on='match_id', right_on='fixture_id', how='inner')
    
    logger.info(f"Successfully mapped {len(df_feat_mapped)} generated features to CSV indexes.")
    
    cols_to_merge = [c for c in df_feat_mapped.columns if c not in ['csv_index', 'match_id', 'fixture_id', 'Date']]
    
    # Drop existing occurrences to prevent _x _y duplicates
    existing_cols = [c for c in cols_to_merge if c in df_csv.columns]
    if existing_cols:
        logger.info(f"Dropping {len(existing_cols)} existing dynamically generated columns to refresh them.")
        df_csv.drop(columns=existing_cols, inplace=True)
    
    # Create the merge key
    df_csv["_row_idx"] = df_csv.index
    df_feat_mapped["csv_index"] = df_feat_mapped["csv_index"].astype(int)
    
    merged = df_csv.merge(
        df_feat_mapped[['csv_index'] + cols_to_merge],
        left_on="_row_idx",
        right_on="csv_index",
        how="left"
    )
    merged.drop(columns=["_row_idx", "csv_index"], inplace=True)
    
    output = csv_path.parent / "features_full_enriched.csv"
    merged.to_csv(output, index=False)
    
    logger.info(f"Merged {len(df_feat_mapped)} rows. Columns: {original_cols} -> {len(merged.columns)}")
    logger.info(f"Saved to {output}. To use: cp {output} {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute Unified Tier 1 Features")
    parser.add_argument("--merge", action="store_true", help="Merge output to features_full.csv")
    args = parser.parse_args()
    
    logger.info("Loading Datasets...")
    df_enriched = pd.read_parquet(ENRICHED_PATH)
    df_full = pd.read_csv(FEATURES_CSV, low_memory=False)
    
    # Sync match outcome columns for parsing
    stat_cols = ['FTHG', 'FTAG', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']
    stat_cols = [c for c in stat_cols if c in df_full.columns]
    
    df_full_idx = df_full[stat_cols].reset_index(names='csv_index')
    df_merged = df_enriched.merge(df_full_idx, on='csv_index', how='left')
    
    logger.info(f"Merged Parquet Length: {len(df_merged)}")
    
    pipe = DailyPipeline()
    logger.info("Parsing to raw Pydantic Matches (this will take 30-60 seconds)...")
    matches = pipe._df_to_matches(df_merged, is_historical=True)
    
    completed = [m for m in matches if m.is_completed]
    logger.info(f"Total structured matches derived: {len(completed)} / {len(matches)}")
    
    logger.info("Running unified FeatureRegistry offline trajectory...")
    registry = FeatureRegistry(training_mode=False)
    
    df_features = registry.transform_historical(completed)
    
    df_features.to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved primary feature array to {OUTPUT_PATH}")
    
    if args.merge:
        merge_into_csv(df_features, df_enriched, FEATURES_CSV)

    logger.info("✅ Done!")

if __name__ == "__main__":
    main()
