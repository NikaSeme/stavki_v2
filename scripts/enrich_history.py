#!/usr/bin/env python3
"""
Enrich Historical Data
======================

Backfills xG and Lineup data for historical matches.
required for Phase 2 (Advanced & Roster features).

Usage:
    python3 scripts/enrich_history.py --limit 100
"""

import sys
import logging
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.config import get_config
from stavki.data.loader import get_loader
from stavki.data.collectors.sportmonks import SportMonksCollector
from stavki.data.schemas import Match, Team, League

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("enricher")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max matches to process (0=all)")
    parser.add_argument("--output", default="data/features_enriched.parquet")
    args = parser.parse_args()
    
    config = get_config()
    if not config.sportmonks_api_key:
        logger.error("SportMonks API key required!")
        sys.exit(1)
        
    loader = get_loader()
    collector = SportMonksCollector()
    
    # 1. Load existing history
    logger.info("Loading history...")
    df = loader.get_historical_data()
    logger.info(f"Loaded {len(df)} matches")
    
    # Check if we have rich columns already
    output_path = PROJECT_ROOT / args.output
    if output_path.exists():
        logger.info(f"Loading existing enrichment from {args.output}")
        rich_df = pd.read_parquet(output_path)
        # Merge or resume?
        # Ideally we want to process rows in `df` that are NOT in `rich_df` or have missing info
        # For simplicity, let's assume `rich_df` tracks processed ID
        processed_ids = set(rich_df["match_id"]) if "match_id" in rich_df.columns else set()
    else:
        rich_df = pd.DataFrame()
        processed_ids = set()
        
    # 2. Filter for matches needing enrichment
    # We need fixture_ids. If df has 'fixture_id', use it.
    if "fixture_id" not in df.columns:
        logger.error("Dataframe missing 'fixture_id'. Cannot fetch details.")
        return
        
    matches_to_process = []
    
    for _, row in df.iterrows():
        # Check if fixture_id exists and is valid
        if "fixture_id" not in row or pd.isna(row["fixture_id"]):
            continue
            
        # Check if match is completed
        if "FTR" not in row or pd.isna(row["FTR"]):
            continue
            
        fid = str(int(row["fixture_id"])) # Ensure int string
        if fid in processed_ids:
            continue
        matches_to_process.append(row)
        
    logger.info(f"Found {len(matches_to_process)} matches with IDs to enrich")
        
    logger.info(f"Found {len(matches_to_process)} matches to enrich")
    
    if args.limit > 0:
        matches_to_process = matches_to_process[:args.limit]
        
    # 3. Process
    new_rows = []
    
    try:
        for row in tqdm(matches_to_process):
            fid = str(row["fixture_id"])
            
            # Construct minimal Match object for collector
            match = Match(
                id=fid,
                home_team=Team(name=row["HomeTeam"]),
                away_team=Team(name=row["AwayTeam"]),
                league=League.EPL, # Dummy, not used by fetch_details
                commence_time=datetime.now(), # Dummy
                source="sportmonks",
                source_id=fid
            )
            
            stats, lineups = collector.fetch_match_details(match)
            
            # Flatten data
            rich_row = row.to_dict()
            rich_row["match_id"] = fid
            rich_row["enriched_at"] = datetime.now()
            
            if stats:
                rich_row["xg_home"] = stats.xg_home
                rich_row["xg_away"] = stats.xg_away
                rich_row["home_shots"] = stats.shots_home
                rich_row["away_shots"] = stats.shots_away
                rich_row["home_possession"] = stats.possession_home
                rich_row["away_possession"] = stats.possession_away
                
            if lineups:
                # Store lineups as JSON string
                if lineups.home:
                    rich_row["home_lineup"] = lineups.home.model_dump_json()
                if lineups.away:
                    rich_row["away_lineup"] = lineups.away.model_dump_json()
                    
            new_rows.append(rich_row)
            
            # Rate limit politeness (client handles it, but good to be safe)
            # 500ms delay
            # time.sleep(0.1)
            
            # Periodic save
            if len(new_rows) >= 50:
                _save_chunk(new_rows, output_path, rich_df)
                new_rows = []
                # Reload rich_df to keep state
                if output_path.exists():
                    rich_df = pd.read_parquet(output_path)
                    
    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving progress...")
        
    # Final save
    if new_rows:
        _save_chunk(new_rows, output_path, rich_df)
        
    logger.info("Done.")

def _save_chunk(new_rows, path, existing_df):
    new_df = pd.DataFrame(new_rows)
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
        
    # Deduplicate by match_id, keep last
    combined = combined.drop_duplicates(subset=["match_id"], keep="last")
    
    combined.to_parquet(path, index=False)
    logger.info(f"Saved {len(combined)} rows to {path}")

if __name__ == "__main__":
    main()
