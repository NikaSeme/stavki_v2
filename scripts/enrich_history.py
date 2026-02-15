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
    total_processed = 0
    
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
            
            # Fetch details
            try:
                stats, lineups = collector.fetch_match_details(match)
            except Exception as e:
                logger.warning(f"Failed to fetch details for {fid}: {e}")
                stats, lineups = None, None
            
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
            total_processed += 1
            
            # Periodic save (write to temp part)
            if len(new_rows) >= 500: # Increased chunk size
                _write_temp_chunk(new_rows, output_path)
                new_rows = [] 
                
    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving remaining progress...")
        
    # Final save of buffer
    if new_rows:
        _write_temp_chunk(new_rows, output_path)
        
    # Merge all temp files
    _finalize_merge(output_path)
        
    logger.info(f"Done. Processed {total_processed} matches.")

def _write_temp_chunk(new_rows, main_path):
    """Write a standalone temporary chunk."""
    if not new_rows:
        return
        
    # Unique temp name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_path = main_path.parent / f"{main_path.stem}_temp_{timestamp}.parquet"
    
    df = pd.DataFrame(new_rows)
    df.to_parquet(temp_path, index=False)
    # logger.debug(f"Wrote temp chunk: {temp_path.name}") # Verbose

def _finalize_merge(main_path):
    """Merge all temp chunks into the main file."""
    import glob
    
    # Find all temp files matching pattern
    pattern = f"{main_path.parent}/{main_path.stem}_temp_*.parquet"
    temp_files = sorted(glob.glob(pattern))
    
    if not temp_files:
        logger.info("No new data chunks to merge.")
        return
        
    logger.info(f"Merging {len(temp_files)} temporary chunks...")
    
    # Load all temps
    dfs = []
    
    # Load existing main file if it exists
    if main_path.exists():
        try:
            dfs.append(pd.read_parquet(main_path))
        except Exception as e:
            logger.error(f"CRITICAL: Could not read main file {main_path}: {e}")
            logger.error("Aborting merge to prevent data loss. Temp files preserved.")
            raise e
            
    # Load chunks
    for tf in temp_files:
        try:
            dfs.append(pd.read_parquet(tf))
        except Exception as e:
            logger.error(f"Corrupt temp file {tf}: {e}")
            
    if not dfs:
        return
        
    # Concat
    combined = pd.concat(dfs, ignore_index=True)
    
    # Deduplicate
    combined = combined.drop_duplicates(subset=["match_id"], keep="last")
    
    # Write optimized result
    combined.to_parquet(main_path, index=False)
    logger.info(f"Successfully merged into {main_path} (Total rows: {len(combined)})")
    
    # Cleanup temps
    for tf in temp_files:
        try:
            Path(tf).unlink()
        except:
            pass
    logger.info("Cleaned up temp files.")

if __name__ == "__main__":
    main()
