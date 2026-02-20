
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def build_context():
    silver_path = PROJECT_ROOT / "data" / "processed" / "matches" / "match_events_silver.parquet"
    if not silver_path.exists():
        logger.error(f"Silver events not found at {silver_path}")
        return

    logger.info("Loading Silver Events...")
    df = pd.read_parquet(silver_path)
    # Ensure date proper type
    df['date'] = pd.to_datetime(df['date'])
    
    # We need to melt to Team-Match level to calculate rolling stats
    # Current rows: match_id, home_id, away_id, home_reds, away_reds...
    
    home_cols = ['match_id', 'date', 'home_team_id', 'home_red_cards', 'home_injuries', 'home_penalties', 'home_own_goals', 'var_interventions', 'home_own_goals'] 
    away_cols = ['match_id', 'date', 'away_team_id', 'away_red_cards', 'away_injuries', 'away_penalties', 'away_own_goals', 'var_interventions', 'away_own_goals']
    
    # Rename to generic
    generic_cols = ['match_id', 'date', 'team_id', 'red_cards', 'injuries', 'penalties', 'own_goals', 'var', 'clumsiness']
    
    # Home Split
    h_df = df.copy()
    h_df = h_df.rename(columns={
        'home_team_id': 'team_id',
        'home_red_cards': 'red_cards',
        'home_injuries': 'injuries',
        'home_penalties': 'penalties', # Attacking Pens? We mined "converts" mostly. Let's call it 'penalties'
        'home_own_goals': 'own_goals',
        'var_interventions': 'var'
    })
    h_df = h_df[['match_id', 'date', 'team_id', 'red_cards', 'injuries', 'penalties', 'own_goals', 'var']]
    
    # Away Split
    a_df = df.copy()
    a_df = a_df.rename(columns={
        'away_team_id': 'team_id',
        'away_red_cards': 'red_cards',
        'away_injuries': 'injuries',
        'away_penalties': 'penalties',
        'away_own_goals': 'own_goals',
        'var_interventions': 'var'
    })
    a_df = a_df[['match_id', 'date', 'team_id', 'red_cards', 'injuries', 'penalties', 'own_goals', 'var']]
    
    # Concat
    matches_long = pd.concat([h_df, a_df], ignore_index=True)
    matches_long = matches_long.sort_values(['team_id', 'date'])
    
    logger.info(f"Processing {len(matches_long)} team-match records...")
    
    # Rolling Features
    cols_to_roll = ['red_cards', 'injuries', 'penalties', 'own_goals', 'var']
    
    for col in cols_to_roll:
        # Sum last 10 matches
        matches_long[f'{col}_last10'] = matches_long.groupby('team_id')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).sum().shift(1)
        ).fillna(0)
        
    # Select columns
    final_cols = ['match_id', 'team_id'] + [f'{c}_last10' for c in cols_to_roll]
    out_df = matches_long[final_cols].copy()
    
    # Save
    out_path = PROJECT_ROOT / "data" / "processed" / "teams" / "context_features_gold.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_df.to_parquet(out_path)
    logger.info(f"Saved {len(out_df)} context vectors to {out_path}")
    print(out_df.tail())

if __name__ == "__main__":
    build_context()
