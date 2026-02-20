import os
import sys
import glob
import gzip
import json
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stavki.config import get_config, DATA_DIR

logger = logging.getLogger("process_managers")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

config = get_config()

def process_managers():
    """Extract manager/coach data into managers_silver.parquet."""
    logger.info("Starting Manager/Coach data extraction...")
    raw_dir = DATA_DIR / "raw" / "fixtures"
    output_dir = DATA_DIR / "matches"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "managers_silver.parquet"
    
    files = list(raw_dir.rglob("*.json.gz"))
    if not files:
        logger.info("No raw JSON fixtures found to process. Returning.")
        return
        
    logger.info(f"Processing {len(files)} matches for Manager Encodings...")
    
    records = []
    
    for file_path in tqdm(files, desc="Parsing Managers"):
        try:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
            
            fixture_id = data.get("id")
            if not fixture_id:
                continue
                
            # Coach data might be injected in 'coaches' due to API payload
            coaches = data.get("coaches", [])
            home_coach_id = None
            away_coach_id = None
            
            for coach in coaches:
                meta = coach.get("meta", {})
                location = meta.get("location")
                coach_id = coach.get("coach_id")
                
                if location == "home":
                    home_coach_id = coach_id
                elif location == "away":
                    away_coach_id = coach_id
            
            records.append({
                "match_id": fixture_id,
                "home_coach_id": home_coach_id,
                "away_coach_id": away_coach_id
            })
            
        except Exception as e:
            logger.debug(f"Error parsing file {file_path}: {e}")
            
    df = pd.DataFrame(records)
    logger.info(f"Processed {len(df)} manager records.")
    
    if not df.empty:
        df.to_parquet(out_path)
        unique_managers = pd.concat([df["home_coach_id"], df["away_coach_id"]]).dropna().unique()
        logger.info(f"Saved to {out_path}.")
        logger.info(f"Total Unique Tactical Managers Identified: {len(unique_managers)}")

if __name__ == "__main__":
    process_managers()
