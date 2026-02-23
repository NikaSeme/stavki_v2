#!/usr/bin/env python3
import sys
import os
import json
import gzip
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("append_daily")

def main():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    csv_path = PROJECT_ROOT / "data" / "features_full.csv"
    
    if not csv_path.exists():
        logger.error("features_full.csv not found!")
        sys.exit(1)
        
    df_full = pd.read_csv(csv_path, low_memory=False)
    df_full['Date'] = pd.to_datetime(df_full['Date'], errors='coerce')
    max_date = df_full['Date'].max()
    logger.info(f"Current features_full.csv max date: {max_date}")
    
    league_map = {
        8: "Premier League",
        9: "Championship",
        82: "Bundesliga",
        384: "Serie A",
        564: "La Liga",
        301: "Ligue 1"
    }
    
    new_rows = []
    
    from stavki.data.processors.normalize import TeamMapper
    mapper = TeamMapper()
    
    for gz_path in raw_dir.glob("*.json.gz"):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception:
                continue
                
        # Must be completed
        state = data.get('state', {}).get('short_name', '')
        if state not in ['FT', 'AET', 'PEN']:
            continue
            
        date_str = data['starting_at'].split()[0]
        match_date = pd.to_datetime(date_str)
        
        # Only process matches strictly strictly newer than max_date
        # or after 2026-02-15 to be safe
        if match_date < pd.to_datetime("2026-02-16"):
            continue
            
        # Check if already exists in df_full by date + team
        
        league_id = data.get('league_id')
        league_name = league_map.get(league_id, "Unknown")
        
        home_name = None
        away_name = None
        
        participants = data.get('participants', [])
        for p in participants:
            meta = p.get('meta', {})
            loc = meta.get('location', 'home')
            if loc == 'home':
                home_name = p['name']
            elif loc == 'away':
                away_name = p['name']
                
        if not home_name or not away_name:
            continue
            
        # Standardize using mapped names to match features_full
        home_m = mapper.map_name(home_name) or home_name
        away_m = mapper.map_name(away_name) or away_name
        
        # Avoid duplicate appending natively
        exists = df_full[(df_full['Date'] == match_date) & 
                        ((df_full['HomeTeam'].apply(lambda x: mapper.map_name(str(x)) or str(x)) == home_m) | 
                         (df_full['AwayTeam'].apply(lambda x: mapper.map_name(str(x)) or str(x)) == away_m))]
        if not exists.empty:
            continue
            
        # Parse scores
        scores = data.get('scores', [])
        fthg, ftag = None, None
        
        # Identify home and away IDs
        home_id, away_id = None, None
        for p in participants:
            if p.get('meta', {}).get('location') == 'home': home_id = p['id']
            if p.get('meta', {}).get('location') == 'away': away_id = p['id']
            
        for s in scores:
            if s.get('description') == 'CURRENT':
                if s.get('participant_id') == home_id:
                    fthg = s['score']['goals']
                elif s.get('participant_id') == away_id:
                    ftag = s['score']['goals']
                    
        if fthg is None or ftag is None:
            continue
            
        row = {
            'Date': match_date,
            'HomeTeam': home_name, # Map to canonical representation so model handles it
            'AwayTeam': away_name,
            'FTHG': fthg,
            'FTAG': ftag,
            'League': league_name,
            'Div': next((k for k, v in {"E0":8, "E1":9, "D1":82, "I1":384, "SP1":564, "F1":301}.items() if v == league_id), "")
        }
        
        stats = data.get('statistics', [])
        for stat in stats:
            pid = stat.get('participant_id')
            prefix = 'H' if pid == home_id else 'A'
            t = stat.get('type', {}).get('code', '')
            val = stat.get('data', {}).get('value', 0)
            
            if t == 'shots': row[f'{prefix}S'] = val
            elif t == 'shots-on-target': row[f'{prefix}ST'] = val
            elif t == 'corners': row[f'{prefix}C'] = val
            elif t == 'fouls': row[f'{prefix}F'] = val
            elif t == 'yellow-cards': row[f'{prefix}Y'] = val
            elif t == 'red-cards': row[f'{prefix}R'] = val
            
        new_rows.append(row)
        
    if not new_rows:
        logger.info("No new matches to append to features_full.csv")
        return
        
    df_new = pd.DataFrame(new_rows)
    logger.info(f"Appending {len(df_new)} new matches to features_full.csv")
    
    # Restore Date back to string for CSV
    df_new['Date'] = df_new['Date'].dt.strftime('%Y-%m-%d')
    df_full['Date'] = df_full['Date'].dt.strftime('%Y-%m-%d')
    
    # Concat
    df_combined = pd.concat([df_full, df_new], ignore_index=True)
    
    # Write safe backup 
    backup_path = csv_path.with_suffix('.csv.bak_append')
    if not backup_path.exists():
        df_full.to_csv(backup_path, index=False)
        
    df_combined.to_csv(csv_path, index=False)
    logger.info("Successfully appended to features_full.csv")
    
if __name__ == "__main__":
    main()
