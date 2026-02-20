
import sys
import json
import gzip
import time
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stavki.data.collectors.sportmonks import SportMonksClient
from stavki.config import PROJECT_ROOT

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("harvest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# USER API KEY (The "Gold" Key)
API_KEY = "wFzsoRjY2uZfEpMhUDPxwLBw9o7JJVIZi1AjolrzN7Wqda0OjeTh32XAgZOB"

def compress_json_save(data: dict, path: Path):
    """Save dictionary as compressed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='UTF-8') as f:
        json.dump(data, f)

def harvest_lineups(limit: int = 0):
    """
    Harvest full match data for historical fixtures.
    
    Args:
        limit (int): Max number of matches to fetch (0 for all).
    """
    client = SportMonksClient(api_key=API_KEY)
    
    # 1. Load Match Map
    map_path = PROJECT_ROOT / "data" / "fixture_id_map.json"
    if not map_path.exists():
        logger.error(f"Map not found at {map_path}")
        return
        
    logger.info("Loading fixture map...")
    with open(map_path) as f:
        fixture_map = json.load(f)
    
    # Create targets list
    targets = []
    
    logger.info(f"Loaded map type: {type(fixture_map)}")
    
    # Check for 'matches' key (common pattern)
    if isinstance(fixture_map, dict) and 'matches' in fixture_map:
        logger.info("Found 'matches' key, using it.")
        fixture_map = fixture_map['matches']
        
    if isinstance(fixture_map, dict):
        # Iterate and find valid match objects
        for k, v in fixture_map.items():
            if isinstance(v, dict) and 'fixture_id' in v:
                targets.append(v)
                
    elif isinstance(fixture_map, list):
         for v in fixture_map:
             if isinstance(v, dict) and 'fixture_id' in v:
                 targets.append(v)

    # Sort by Date Descending (Newest First)
    logger.info("Sorting targets by date (newest first)...")
    try:
        targets.sort(key=lambda x: x.get('date', '1900-01-01'), reverse=True)
    except Exception as e:
        logger.warning(f"Sort failed: {e}. Proceeding unsorted.")
    
    logger.info(f"Found {len(targets)} potential targets.")
    
    # 2. Identify Missing Data
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    to_fetch = []
    
    for t in targets:
        fid = t['fixture_id']
        lid = t.get('league_id', 'unknown')
        
        # Check if exists
        save_path = raw_dir / str(lid) / f"{fid}.json.gz"
        if not save_path.exists():
            to_fetch.append((fid, lid, save_path))
            
    logger.info(f"Resuming harvest: {len(to_fetch)} matches missing out of {len(targets)}.")
    
    if limit > 0:
        to_fetch = to_fetch[:limit]
        logger.info(f"Limit applied. Fetching {len(to_fetch)} matches.")
        
    # 3. Harvest Loop
    success_count = 0
    error_count = 0
    
    includes = [
        "sidelined.sideline.player", # Injuries (Gold)
        "sidelined.sideline.type",
        "lineups.details.type",      # Player Stats (Silver)
        "lineups.player",
        "statistics.type",           # Match Stats + Types
        "events.type",               # Events + Types
        "referees.referee",
        "state",
        "venue",
        "participants",
        "league",
        "trends",                    # Validated: Success
        "comments",                  # Validated: Success
        "scores"                     # Validated: critical for outcome
    ]
    
    pbar = tqdm(to_fetch, desc="Harvesting")
    
    for fid, lid, save_path in pbar:
        try:
            # Rate limit buffer (client handles 429s, but let's be nice)
            # time.sleep(0.2) 
            
            response = client._request(f"fixtures/{fid}", includes=includes)
            
            if response.get("error"):
                logger.error(f"API Error for {fid}: {response['error']}")
                error_count += 1
                continue
                
            data = response.get("data")
            if not data:
                logger.warning(f"No data for {fid}")
                error_count += 1
                continue
                
            # Quick validation check for critical data
            has_injuries = len(data.get('sidelined', [])) > 0
            has_details = False
            lineups = data.get('lineups', [])
            if lineups and lineups[0].get('details'):
                has_details = True
                
            # Save Raw Bronze Data
            compress_json_save(data, save_path)
            
            success_count += 1
            pbar.set_postfix({"✅": success_count, "❌": error_count, "🚑": has_injuries, "👕": has_details})
            
        except Exception as e:
            logger.error(f"Exception fetching {fid}: {e}")
            error_count += 1
            # Basic backoff
            time.sleep(1)

    logger.info(f"Harvest complete. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    # Default limit to 10 for testing, user can remove it
    limit_arg = 10
    if len(sys.argv) > 1:
        limit_arg = int(sys.argv[1])
    harvest_lineups(limit=limit_arg)
