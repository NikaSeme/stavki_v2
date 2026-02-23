
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
    
    # Calculate positional median to fill missing ratings safely instead of arbitrary 6.0
    position_medians = df.groupby('position_id')['rating_last_5'].transform('median')
    df['rating_last_5'] = df['rating_last_5'].fillna(position_medians).fillna(6.0)
    
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
    xi_stats = starters.groupby(['match_id', 'team_id']).agg(
        xi_rating=('rating_last_5', 'mean'),
        xi_rating_max=('rating_last_5', 'max'),
        xi_rating_min=('rating_last_5', 'min'),
        xi_rating_var=('rating_last_5', 'var'),
        xi_consistency=('rating_std_10', 'mean'),  # Consistency of XI
        xi_fatigue=('minutes_last_5', lambda x: x.mean() / 90.0), # Fatigue
        xi_goals_form=('goals_last_10', 'sum'),    # Firepower
        xi_creativity=('key_passes_last_10', 'sum'), # Creativity
        xi_class=('career_rating', 'mean')         # Experience/Class
    )
    
    # Fill variance NaNs with 0 (in case of single player XI records)
    xi_stats['xi_rating_var'] = xi_stats['xi_rating_var'].fillna(0.0)
    
    # 3. Calculate Bench Impact (Split Positional)
    # Defense/GK positions: 24 (GK), 25 (Def)
    # Offense/Mid positions: 26 (Mid), 27 (Att)
    bench['is_defensive'] = bench['position_id'].isin([24, 25])
    bench['is_offensive'] = bench['position_id'].isin([26, 27])
    
    # Top 2 Defensive Bench Players
    bench_def = bench[bench['is_defensive']].sort_values('rating_last_5', ascending=False)
    bench_def_top2 = bench_def.groupby(['match_id', 'team_id']).head(2)
    bench_def_stats = bench_def_top2.groupby(['match_id', 'team_id']).agg(
        bench_def_strength=('rating_last_5', 'mean')
    )
    
    # Top 2 Offensive Bench Players
    bench_off = bench[bench['is_offensive']].sort_values('rating_last_5', ascending=False)
    bench_off_top2 = bench_off.groupby(['match_id', 'team_id']).head(2)
    bench_off_stats = bench_off_top2.groupby(['match_id', 'team_id']).agg(
        bench_off_strength=('rating_last_5', 'mean')
    )
    
    # 4. Merge
    team_vectors = xi_stats.join(bench_def_stats, how='left').join(bench_off_stats, how='left')
    
    # Fill missing bench with average rating 6.0
    team_vectors['bench_def_strength'] = team_vectors['bench_def_strength'].fillna(6.0)
    team_vectors['bench_off_strength'] = team_vectors['bench_off_strength'].fillna(6.0)
    
    # Save
    out_path = PROJECT_ROOT / "data" / "processed" / "teams" / "team_vectors_gold.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    team_vectors.to_parquet(out_path)
    logger.info(f"Saved {len(team_vectors)} team vectors to {out_path}")
    print(team_vectors.head())

if __name__ == "__main__":
    build_team_vectors()
