#!/usr/bin/env python3
"""
Fetch Daily Results (Online Learning Ingestion)
=============================================

This script connects to the SportMonks API to download the actual outcomes
of matches from the previous 48 hours. It saves the raw JSON payloads
to the systematic datalake (`data/raw/matches`), allowing the pipeline
to rebuild the feature vectors and provide ground-truth targets for Continual Learning.

Usage:
    python scripts/fetch_daily_results.py
    python scripts/fetch_daily_results.py --days-back 3
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
import gzip
import json

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.config import get_config
from stavki.data.collectors.sportmonks import SportMonksClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("DailyResultsScraper")

def main():
    parser = argparse.ArgumentParser(description="Fetch recent match results from SportMonks")
    parser.add_argument("--days-back", type=int, default=2, help="Number of days to look back")
    args = parser.parse_args()

    config = get_config()
    if not config.sportmonks_api_key:
        logger.error("SPORTMONKS_API_KEY is not set in environment or .env file.")
        sys.exit(1)

    client = SportMonksClient(api_key=config.sportmonks_api_key)
    
    # Calculate Date Range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days_back)
    
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    logger.info("=" * 50)
    logger.info(f"Fetching Match Results from {start_str} to {end_str}")
    logger.info("=" * 50)

    total_saved = 0
    
    # Extract League IDs from config map
    all_leagues = []
    for k, v in client.league_ids.items():
        if isinstance(v, int) and v not in all_leagues:
            all_leagues.append(v)
            
    logger.info(f"Monitoring targeted leagues: {all_leagues}")
    
    # Ensure raw data directory exists
    matches_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    matches_dir.mkdir(parents=True, exist_ok=True)

    try:
        # We need the RAW json payloads complete with statistics and lineups 
        # to cleanly rebuild the feature vectors. We will bypass the MatchFixture 
        # parser and utilize client._request natively spanning the specific dates.
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info(f"Scanning fixtures for date: {date_str}")
            
            page = 1
            endpoint = f"fixtures/date/{date_str}"
            
            while True:
                params = {"page": str(page)}
                
                response = client._request(
                    endpoint,
                    params=params,
                    includes=["state", "participants", "scores", "events", "lineups", "statistics", "trends"]
                )
                
                data_items = response.get("data", [])
                if not data_items:
                    break
                
                for fixture in data_items:
                    # Filter to only approved leagues to save disk space
                    league_id = fixture.get("league_id")
                    if league_id not in all_leagues:
                        continue
                        
                    # We ONLY want matches that have actually finished to get the final score
                    status_obj = fixture.get("state", {})
                    status = status_obj.get("short_name", status_obj.get("state", "NS")).upper()
                    
                    if status not in ["FT", "AET", "PEN", "AWB"]:
                        continue
                        
                    fixture_id = fixture.get("id")
                    if not fixture_id:
                        continue
                        
                    # Serialize the payload cleanly to disk exactly like historical batch jobs
                    payload = json.dumps(fixture, indent=2)
                    filepath = matches_dir / f"{fixture_id}.json.gz"
                    
                    with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                        f.write(payload)
                        
                    total_saved += 1
                
                # Check pagination bounds
                pagination = response.get("pagination", {})
                if not pagination.get("has_more", False):
                    break
                page += 1
                
            current_date += timedelta(days=1)
            time.sleep(1) # Be gentle to API limits doing raw crawls
            
        logger.info(f"Successfully saved {total_saved} completed match vectors to {matches_dir.relative_to(PROJECT_ROOT)}")

    except Exception as e:
        logger.error(f"Failed to fetch daily results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    logger.info("Daily Results Fetch Complete.")

if __name__ == "__main__":
    main()
