
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

def build_features():
    silver_path = PROJECT_ROOT / "data" / "processed" / "players" / "player_stats_silver.parquet"
    if not silver_path.exists():
        logger.error(f"Silver data not found at {silver_path}")
        return

    logger.info("Loading Silver Player Data...")
    df = pd.read_parquet(silver_path)
    logger.info(f"Loaded {len(df)} rows.")

    # Sort for rolling calcs
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['player_id', 'date'])

    # Fill NaNs in Rating with 6.0 (Average/Base performance)
    df['rating'] = df['rating'].fillna(6.0)
    df['minutes'] = df['minutes'].fillna(0)
    
    # --- Feature Engineering ---
    
    logger.info("Calculating Rolling Features...")
    
    # Group by Player
    # We use transform to keep original index
    
    # 1. Form (Rating)
    # Rolling 5 matches
    df['rating_last_5'] = df.groupby('player_id')['rating'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1) # Shift 1 to avoid leakage!
    )
    
    # Rolling 10 matches
    df['rating_last_10'] = df.groupby('player_id')['rating'].transform(
        lambda x: x.rolling(window=10, min_periods=3).mean().shift(1)
    )
    
    # Consistency (Std Dev)
    df['rating_std_10'] = df.groupby('player_id')['rating'].transform(
        lambda x: x.rolling(window=10, min_periods=3).std().shift(1)
    ).fillna(0)
    
    # 2. Fatigue (Minutes per day approximation or rolling sum)
    # Rolling sum of minutes last 5 matches (proxy for recent load)
    df['minutes_last_5'] = df.groupby('player_id')['minutes'].transform(
        lambda x: x.rolling(window=5, min_periods=1).sum().shift(1)
    )
    
    # 3. Impact (Goals / Key Passes)
    # Rolling 10 sum
    df['goals_last_10'] = df.groupby('player_id')['goals'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum().shift(1)
    ).fillna(0)
    
    df['key_passes_last_10'] = df.groupby('player_id')['key_passes'].transform(
        lambda x: x.rolling(window=10, min_periods=1).sum().shift(1)
    ).fillna(0)
    
    # 4. Global Averages (Career)
    # Expanding mean (shift 1)
    df['career_rating'] = df.groupby('player_id')['rating'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # Fill remaining NaNs (for first matches of career)
    df['rating_last_5'] = df['rating_last_5'].fillna(6.0)
    df['rating_last_10'] = df['rating_last_10'].fillna(6.0)
    df['career_rating'] = df['career_rating'].fillna(6.0)
    df = df.fillna(0) # Rest (minutes, goals) are 0
    
    # Save Gold
    gold_path = PROJECT_ROOT / "data" / "processed" / "players" / "player_features_gold.parquet"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select feature columns + keys
    cols = [
        'match_id', 'date', 'player_id', 'team_id', 'is_home', 
        'position_id', 'formation_position', 'lineup_type_id',
        'rating_last_5', 'rating_last_10', 'rating_std_10', 'career_rating',
        'minutes_last_5', 'goals_last_10', 'key_passes_last_10'
    ]
    
    out_df = df[cols]
    out_df.to_parquet(gold_path)
    logger.info(f"Saved {len(out_df)} rows to {gold_path}")
    logger.info("Sample:")
    print(out_df.tail())

if __name__ == "__main__":
    build_features()
