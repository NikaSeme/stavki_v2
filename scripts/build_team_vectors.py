
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

def build_team_vectors():
    gold_path = PROJECT_ROOT / "data" / "processed" / "players" / "player_features_gold.parquet"
    if not gold_path.exists():
        logger.error(f"Gold features not found at {gold_path}")
        return

    logger.info("Loading Gold Player Features...")
    df = pd.read_parquet(gold_path)
    
    # Fill NaN ratings with 6.0
    df['rating_last_5'] = df['rating_last_5'].fillna(6.0)
    
    # --- Aggregation Logic ---
    
    # 1. Split Starters vs Bench
    # Type 11 = Start XI, 12 = Bench
    # If missing, assume top 11 by rating are starters? (Fallback)
    # But we trust our harvester.
    
    starters = df[df['lineup_type_id'] == 11].copy()
    bench = df[df['lineup_type_id'] == 12].copy()
    
    logger.info(f"Starters: {len(starters)}, Bench: {len(bench)}")
    
    # 2. Calculate XI Features
    # Group by match_id, team_id
    xi_stats = starters.groupby(['match_id', 'team_id']).agg({
        'rating_last_5': 'mean',
        'rating_std_10': 'mean', # Consistency of XI
        'minutes_last_5': lambda x: x.mean() / 90.0, # Fatigue (Normalized to Matches played)
        'goals_last_10': 'sum',   # Firepower
        'key_passes_last_10': 'sum', # Creativity
        'career_rating': 'mean'   # Experience/Class
    }).rename(columns={
        'rating_last_5': 'xi_rating',
        'rating_std_10': 'xi_consistency',
        'minutes_last_5': 'xi_fatigue',
        'goals_last_10': 'xi_goals_form',
        'key_passes_last_10': 'xi_creativity',
        'career_rating': 'xi_class'
    })
    
    # 3. Calculate Bench Impact
    # Take top 5 bench players by rating (impact subs)
    bench_sorted = bench.sort_values('rating_last_5', ascending=False)
    bench_top5 = bench_sorted.groupby(['match_id', 'team_id']).head(5)
    
    bench_stats = bench_top5.groupby(['match_id', 'team_id']).agg({
        'rating_last_5': 'mean' 
    }).rename(columns={'rating_last_5': 'bench_strength'})
    
    # 4. Merge
    team_vectors = xi_stats.join(bench_stats, how='left')
    
    # Fill missing bench (if no bench data) with average rating 6.0
    team_vectors['bench_strength'] = team_vectors['bench_strength'].fillna(6.0)
    
    # Save
    out_path = PROJECT_ROOT / "data" / "processed" / "teams" / "team_vectors_gold.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    team_vectors.to_parquet(out_path)
    logger.info(f"Saved {len(team_vectors)} team vectors to {out_path}")
    print(team_vectors.head())

if __name__ == "__main__":
    build_team_vectors()
