
import sys
import json
import gzip
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_trends():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    files = list(raw_dir.rglob("*.json.gz"))
    
    logger.info(f"Found {len(files)} raw fixture files.")
    
    trend_rows = []
    
    for fpath in tqdm(files, desc="Processing Trends"):
        try:
            with gzip.open(fpath, 'rt', encoding='UTF-8') as f:
                data = json.load(f)
                
            fid = data.get('id')
            # Check for trends
            trends = data.get('trends', [])
            if not trends:
                continue
                
            # Participants to map team_id
            participants = data.get('participants', [])
            home = next((p for p in participants if p['meta']['location'] == 'home'), {})
            away = next((p for p in participants if p['meta']['location'] == 'away'), {})
            
            home_id = home.get('id')
            away_id = away.get('id')
            
            for t in trends:
                row = {
                    'match_id': fid,
                    'team_id': t.get('participant_id'), # Might be team ID directly?
                    'minute': t.get('minute'),
                    'type_id': t.get('type_id'),
                    'value': t.get('value')
                }
                # Check mapping
                # SportMonks trends usually have participant_id as team_id
                
                trend_rows.append(row)
                
        except Exception as e:
            logger.error(f"Error processing {fpath}: {e}")
            
    if trend_rows:
        df = pd.DataFrame(trend_rows)
        # Sort
        df = df.sort_values(['match_id', 'minute'])
        
        out_path = PROJECT_ROOT / "data" / "processed" / "trends" / "trends_silver.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info(f"Saved {len(df)} trend records to {out_path}")
        print(df.head())
    else:
        logger.warning("No trends found.")

if __name__ == "__main__":
    process_trends()
