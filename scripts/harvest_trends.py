
import sys
import json
import gzip
import time
import logging
from pathlib import Path
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
        logging.FileHandler("harvest_trends.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("HarvestTrends")

API_KEY = "wFzsoRjY2uZfEpMhUDPxwLBw9o7JJVIZi1AjolrzN7Wqda0OjeTh32XAgZOB"

def compress_json_save(data: dict, path: Path):
    """Save dictionary as compressed JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='UTF-8') as f:
        json.dump(data, f)

def harvest_missing_trends(limit: int = 0):
    client = SportMonksClient(api_key=API_KEY)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    
    logger.info("Scanning local json.gz datalake for missing 'trends' arrays...")
    all_files = list(raw_dir.rglob("*.json.gz"))
    
    to_fetch = []
    
    for fpath in tqdm(all_files, desc="Scanning files"):
        needs_fetch = False
        try:
            with gzip.open(fpath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
                if not data.get("trends"):
                    needs_fetch = True
        except Exception:
            needs_fetch = True
            
        if needs_fetch:
            # Reconstruct lid and fid from path
            # path is .../data/raw/fixtures/{league_id}/{fixture_id}.json.gz
            fid = fpath.stem.replace(".json", "")
            lid = fpath.parent.name
            to_fetch.append((fid, lid, fpath))
            
    logger.info(f"Identified {len(to_fetch)} matches missing 'trends' data.")
    
    if limit > 0:
        to_fetch = to_fetch[:limit]
        logger.info(f"Limit applied. Fetching {len(to_fetch)} matches.")

    includes = [
        "sidelined.sideline.player",
        "sidelined.sideline.type",
        "lineups.details.type",
        "lineups.player",
        "statistics.type",
        "events.type",
        "referees.referee",
        "state",
        "venue",
        "participants",
        "league",
        "trends",
        "comments",
        "scores"
    ]

    success_count = 0
    error_count = 0
    
    pbar = tqdm(to_fetch, desc="Harvesting Trends")
    for fid, lid, save_path in pbar:
        try:
            # The client handles rate limiting and 429s automatically
            response = client._request(f"fixtures/{fid}", includes=includes)
            
            if response.get("error"):
                logger.error(f"API Error for {fid}: {response['error']}")
                error_count += 1
                continue
                
            data = response.get("data")
            if not data:
                logger.warning(f"No data returned for {fid}")
                error_count += 1
                continue
                
            has_trends = len(data.get('trends', [])) > 0
            
            compress_json_save(data, save_path)
            
            success_count += 1
            pbar.set_postfix({"✅": success_count, "❌": error_count, "📈": has_trends})
            
        except Exception as e:
            logger.error(f"Exception fetching {fid}: {e}")
            error_count += 1
            time.sleep(1)
            
    logger.info(f"Repair complete. Success: {success_count}, Errors: {error_count}")

if __name__ == "__main__":
    limit_arg = 0
    if len(sys.argv) > 1:
        limit_arg = int(sys.argv[1])
    harvest_missing_trends(limit=limit_arg)
