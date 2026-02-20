
import sys
import json
import gzip
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_stats(details: list) -> dict:
    """Extract key stats from lineup details."""
    stats = {}
    if not details:
        return stats
        
    for item in details:
        type_name = item.get('type', {}).get('code', 'unknown')
        value = item.get('data', {}).get('value')
        
        # Map common stats
        if type_name == 'rating':
            try:
                stats['rating'] = float(value)
            except:
                stats['rating'] = np.nan
        elif type_name == 'minutes-played':
            stats['minutes'] = int(value) if value else 0
        elif type_name == 'goals':
            stats['goals'] = int(value) if value else 0
        elif type_name == 'assists':
            stats['assists'] = int(value) if value else 0
        elif type_name == 'shots-total':
            stats['shots'] = int(value) if value else 0
        elif type_name == 'key-passes':
            stats['key_passes'] = int(value) if value else 0
        elif type_name == 'passes':
            stats['passes'] = int(value) if value else 0
        elif type_name == 'accurate-passes':
            stats['accurate_passes'] = int(value) if value else 0
        elif type_name == 'total-duels':
            stats['total_duels'] = int(value) if value else 0
        elif type_name == 'duels-won':
            stats['duels_won'] = int(value) if value else 0
            
    return stats

def process_files():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    files = list(raw_dir.rglob("*.json.gz"))
    
    logger.info(f"Found {len(files)} raw fixture files.")
    
    player_rows = []
    injury_rows = []
    
    for fpath in tqdm(files, desc="Processing"):
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
            home_id = next((p['id'] for p in participants if p['meta']['location'] == 'home'), None)
            away_id = next((p['id'] for p in participants if p['meta']['location'] == 'away'), None)
            
            # 1. Process Lineups (Player Stats)
            lineups = data.get('lineups', [])
            for p in lineups:
                pid = p.get('player_id')
                tid = p.get('team_id')
                if not pid or not tid:
                    continue
                    
                # Basic info
                row = {
                    'match_id': fid,
                    'date': date_val,
                    'league_id': league_id,
                    'player_id': pid,
                    'team_id': tid,
                    'is_home': tid == home_id,
                    'position_id': p.get('player', {}).get('position_id'), # Natural Position
                    'formation_position': p.get('formation_position'),     # Tactical Slot (1-11)
                    'lineup_type_id': p.get('type_id'),                    # 11=Starter, 12=Bench
                }
                
                # Stats
                stats = parse_stats(p.get('details', []))
                row.update(stats)
                player_rows.append(row)
                
            # 2. Process Injuries (Sidelined)
            sidelined = data.get('sidelined', [])
            for s in sidelined:
                sideline = s.get('sideline', {})
                if not sideline:
                    continue
                    
                i_row = {
                    'match_id': fid, # Recorded at this match time
                    'date': date_val,
                    'player_id': sideline.get('player_id'),
                    'team_id': sideline.get('team_id'),
                    'reason': sideline.get('type', {}).get('name', 'Unknown'),
                    'start_date': sideline.get('start_date'),
                    'end_date': sideline.get('end_date'),
                    'games_missed': sideline.get('games_missed')
                }
                injury_rows.append(i_row)
                
        except Exception as e:
            logger.error(f"Error processing {fpath}: {e}")
            
    # Save Players
    if player_rows:
        df_players = pd.DataFrame(player_rows)
        # Ensure types
        df_players['date'] = pd.to_datetime(df_players['date'])
        out_path = PROJECT_ROOT / "data" / "processed" / "players" / "player_stats_silver.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_players.to_parquet(out_path)
        logger.info(f"Saved {len(df_players)} player stats to {out_path}")
        
    # Save Injuries
    if injury_rows:
        df_injuries = pd.DataFrame(injury_rows)
        out_path = PROJECT_ROOT / "data" / "processed" / "injuries" / "injuries_silver.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_injuries.to_parquet(out_path)
        logger.info(f"Saved {len(df_injuries)} injury records to {out_path}")

if __name__ == "__main__":
    process_files()
