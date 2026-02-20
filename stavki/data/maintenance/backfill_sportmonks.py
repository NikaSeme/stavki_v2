"""
data/maintenance/backfill_sportmonks.py

Backfill historical match data from SportMonks for existing football-data.co.uk matches.

Process:
1. Load all historical matches from `football-data.co.uk` CSVs (via UnifiedDataLoader).
2. For each match:
    a. Check if we already have it in `data/storage/sportmonks_historical.jsonl`.
    b. If not, fetch SportMonks fixtures for that date.
    c. Fuzzy match Home/Away teams to find the correct `fixture_id`.
    d. Fetch detailed stats (xG, lineups, events) for that fixture.
    e. Append to JSONL file.

Usage:
    python -m stavki.data.maintenance.backfill_sportmonks --days 730  # Last 2 years
"""

import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import difflib

from stavki.config import PROJECT_ROOT
from stavki.data.loader import get_loader
from stavki.data.schemas import League

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("backfill_sportmonks.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("backfill")

# Output file
OUTPUT_FILE = PROJECT_ROOT / "stavki" / "data" / "storage" / "sportmonks_historical.jsonl"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

class SportMonksBackfill:
    def __init__(self, days: int = 730, lookback_start: Optional[str] = None):
        self.loader = get_loader()
        self.client = self.loader.client
        self.days = days
        self.lookback_start = lookback_start
        
        if not self.client:
            raise ValueError("SportMonks API key not found!")
            
        # Load existing backfilled data to skip duplicates
        self.existing_ids = set()
        if OUTPUT_FILE.exists():
            with open(OUTPUT_FILE, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # We use a composite key of date+home+away to avoid re-fetching
                        # even if we didn't save the fixture_id previously (though we should have)
                        key = f"{data['date']}_{data['home_team']}_{data['away_team']}"
                        self.existing_ids.add(key)
                    except json.JSONDecodeError:
                        continue
        
        logger.info(f"Loaded {len(self.existing_ids)} existing records.")

    def fuzzy_match_fixture(self, target_home: str, target_away: str, fixtures: List) -> Optional[Dict]:
        """Find the best matching fixture from a list of SportMonks fixtures."""
        best_match = None
        best_score = 0.0
        
        # Normalize local target names
        norm_home = self.loader.normalize_team_name(target_home).lower()
        norm_away = self.loader.normalize_team_name(target_away).lower()
        
        for fix in fixtures:
            # SportMonks names
            sm_home = self.loader.normalize_team_name(fix.home_team).lower()
            sm_away = self.loader.normalize_team_name(fix.away_team).lower()
            
            # Simple containment check first (fast)
            if norm_home == sm_home and norm_away == sm_away:
                return fix
            
            # Fuzzy check
            score_home = difflib.SequenceMatcher(None, norm_home, sm_home).ratio()
            score_away = difflib.SequenceMatcher(None, norm_away, sm_away).ratio()
            avg_score = (score_home + score_away) / 2
            
            if avg_score > best_score:
                best_score = avg_score
                best_match = fix
        
        # Threshold: 0.8 seems safe for team names
        if best_score > 0.8:
            return best_match
        
        return None

    def run(self):
        # 1. Load Target Matches (FD)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        
        if self.lookback_start:
             start_date = datetime.strptime(self.lookback_start, "%Y-%m-%d")
        
        logger.info(f"Loading matches from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        matches_df = self.loader.get_historical_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            leagues=None # All configured leagues
        )
        
        if matches_df.empty:
            logger.warning("No historical matches found in range!")
            return

        logger.info(f"Found {len(matches_df)} matches to process.")
        
        # Cache fixtures per day to avoid re-fetching for every match on the same day
        fixtures_cache = {} 
        
        processed_count = 0
        skip_count = 0
        error_count = 0

        # Group by date to minimize API calls
        matches_by_date = matches_df.groupby('Date')

        for date_val, group in matches_by_date:
            date_str = date_val.strftime("%Y-%m-%d")  # pandas timestamp to str
            
            # Get fixtures for this date (once per day)
            if date_str not in fixtures_cache:
                try:
                    # Fetch for ALL leagues to be safe, filter later matching names
                    # Passing None to league_ids fetches everything (might be huge), 
                    # but we only care about our leagues.
                    # Optimization: Pass our league IDs
                    league_ids = list(self.loader.leagues_map.keys())
                    fixtures_cache[date_str] = self.client.get_fixtures_by_date(date_str, league_ids=league_ids)
                    logger.info(f"Fetched {len(fixtures_cache[date_str])} fixtures for {date_str}")
                    time.sleep(0.5) # Rate limit politeness
                except Exception as e:
                    logger.error(f"Failed to fetch fixtures for {date_str}: {e}")
                    fixtures_cache[date_str] = []
            
            daily_fixtures = fixtures_cache[date_str]
            if not daily_fixtures:
                continue

            for _, row in group.iterrows():
                home_team = row['HomeTeam']
                away_team = row['AwayTeam']
                
                # Check if already processed
                key = f"{date_str}_{home_team}_{away_team}"
                if key in self.existing_ids:
                    skip_count += 1
                    continue
                
                # Find matching SportMonks fixture
                sm_fixture = self.fuzzy_match_fixture(home_team, away_team, daily_fixtures)
                
                if not sm_fixture:
                    logger.warning(f"No match found for {home_team} vs {away_team} on {date_str}")
                    error_count += 1
                    continue
                
                # Fetch FULL details (Lineups, Stats, etc.)
                try:
                    full_details = self.client._request(
                        f"fixtures/{sm_fixture.fixture_id}",
                        includes=["statistics", "lineups", "weatherreport", "participants"]
                    )
                    
                    data = full_details.get("data", {})
                    if not data:
                        logger.warning(f"Empty data for fixture {sm_fixture.fixture_id}")
                        continue

                    # Construct Record
                    # We store the raw-ish sportmonks return, but flattened slightly or keeping structure
                    # Ideally, keep it close to API response so builders parse it naturally
                    
                    record = {
                        "date": date_str,
                        "home_team": home_team, # Our name
                        "away_team": away_team, # Our name
                        "fd_league": row['League'],
                        "sm_fixture_id": sm_fixture.fixture_id,
                        "sm_league_id": sm_fixture.league_id,
                        "sm_data": data # Full nested object
                    }
                    
                    # Append to JSONL
                    with open(OUTPUT_FILE, 'a') as f:
                        f.write(json.dumps(record) + "\n")
                    
                    self.existing_ids.add(key)
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        logger.info(f"Processed {processed_count} matches... (Skips: {skip_count}, Misses: {error_count})")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch details for {sm_fixture.fixture_id}: {e}")
                    error_count += 1
        
        logger.info(f"Backfill Complete. Processed: {processed_count}, Skipped: {skip_count}, Misses: {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill SportMonks Data")
    parser.add_argument("--days", type=int, default=730, help="Number of days to look back")
    parser.add_argument("--start", type=str, help="Specific start date YYYY-MM-DD")
    
    args = parser.parse_args()
    
    backfill = SportMonksBackfill(days=args.days, lookback_start=args.start)
    backfill.run()
