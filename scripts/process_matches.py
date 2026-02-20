
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

def process_matches():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    files = list(raw_dir.rglob("*.json.gz"))
    
    logger.info(f"Found {len(files)} raw fixture files.")
    
    match_rows = []
    
    for fpath in tqdm(files, desc="Processing Matches"):
        try:
            with gzip.open(fpath, 'rt', encoding='UTF-8') as f:
                data = json.load(f)
                
            fid = data.get('id')
            league_id = data.get('league_id')
            
            # Date handling
            start_at = data.get('starting_at')
            if not start_at:
                continue
            date_val = pd.to_datetime(start_at).date()
            
            # Teams
            participants = data.get('participants', [])
            home = next((p for p in participants if p['meta']['location'] == 'home'), {})
            away = next((p for p in participants if p['meta']['location'] == 'away'), {})
            
            home_id = home.get('id')
            away_id = away.get('id')
            
            # Scores (New Include)
            # scores format: [{'score': {'goals': 1, 'participant': 'home'}, 'description': 'CURRENT'}]
            # Or simpler: parse from 'scores' list based on type 'FT' or 'description'
            
            scores = data.get('scores', [])
            home_goals = 0
            away_goals = 0
            valid_score = False
            
            # Try parsing scores object
            if scores:
                for s in scores:
                    desc = s.get('description', '').upper()
                    # Look for FT or 2ND_HALF (if finished)
                    if desc == 'FT' or desc == 'CURRENT': # CURRENT usually means FT if state is Finished
                        score_obj = s.get('score', {})
                        if score_obj.get('participant') == 'home':
                             home_goals = score_obj.get('goals', 0)
                        elif score_obj.get('participant') == 'away':
                             away_goals = score_obj.get('goals', 0)
                        valid_score = True # Potentially partial, need loop
                    
                # The structure might be different: one score per team per type?
                # Actually v3 usually returns a list score objects
                # Let's verify structure with inspector if this fails, but for now apply robust logic
                # Extract FT score specifically
                ft_scores = [s for s in scores if s.get('description') == 'FT']
                if not ft_scores and any(s.get('description') == '2ND_HALF' for s in scores):
                     ft_scores = [s for s in scores if s.get('description') == '2ND_HALF']
                
                if ft_scores:
                     for s in ft_scores:
                        p_type = s.get('score', {}).get('participant')
                        g = s.get('score', {}).get('goals', 0)
                        if p_type == 'home': home_goals = g
                        if p_type == 'away': away_goals = g
                        valid_score = True
            
            # If no scores object, try participants meta (if available/reliable, but earlier check said no score in meta)
            # Fallback to events? No, let's rely on 'scores' include working.
            
            row = {
                'match_id': fid,
                'league_id': league_id,
                'date': date_val,
                'home_team_id': home_id,
                'away_team_id': away_id,
                'home_score': int(home_goals),
                'away_score': int(away_goals),
                'valid_score': valid_score
            }
            
            # Determine outcome
            if valid_score:
                if home_goals > away_goals: row['outcome'] = 0 # Home Win
                elif away_goals > home_goals: row['outcome'] = 2 # Away Win
                else: row['outcome'] = 1 # Draw
            else:
                row['outcome'] = -1
                
            match_rows.append(row)
                
        except Exception as e:
            logger.error(f"Error processing {fpath}: {e}")
            
    if match_rows:
        df = pd.DataFrame(match_rows)
        # Filter invalid
        df = df[df['valid_score'] == True]
        
        out_path = PROJECT_ROOT / "data" / "processed" / "matches" / "matches_silver.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info(f"Saved {len(df)} match records to {out_path}")
        print(df.head())
    else:
        logger.warning("No matches found.")

if __name__ == "__main__":
    process_matches()
