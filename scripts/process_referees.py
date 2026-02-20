"""
Silver layer data processing step.
Extracts match referee assignments from raw fixture JSON files.
"""
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

def process_referees():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    files = list(raw_dir.rglob("*.json.gz"))
    
    logger.info(f"Found {len(files)} raw fixture files.")
    
    match_referees = []
    
    for fpath in tqdm(files, desc="Processing Referees"):
        try:
            with gzip.open(fpath, 'rt', encoding='UTF-8') as f:
                data = json.load(f)
                
            fid = data.get('id')
            if not fid:
                continue
                
            # Refs array from SportMonks
            referees = data.get('referees', [])
            if not referees:
                continue
                
            # Find the main referee (type_id = 6)
            main_ref = next((r for r in referees if r.get('type_id') == 6), None)
            
            if main_ref:
                ref_id = main_ref.get('referee_id')
                ref_obj = main_ref.get('referee', {})
                ref_name = ref_obj.get('common_name', f'Referee_{ref_id}')
                
                if ref_id:
                    match_referees.append({
                        'match_id': fid,
                        'referee_id': int(ref_id),
                        'referee_name': ref_name
                    })
                    
        except Exception as e:
            logger.error(f"Error processing {fpath}: {e}")
            
    if match_referees:
        df = pd.DataFrame(match_referees)
        df = df.drop_duplicates(subset=['match_id'])
        
        out_path = PROJECT_ROOT / "data" / "processed" / "matches" / "referees_silver.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info(f"Saved {len(df)} referee assignments to {out_path}")
        print(f"Unique referees: {df['referee_id'].nunique()}")
        print(df.head())
    else:
        logger.warning("No referee assignments found.")

if __name__ == "__main__":
    process_referees()
