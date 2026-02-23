
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

def build_momentum():
    silver_path = PROJECT_ROOT / "data" / "processed" / "trends" / "trends_silver.parquet"
    if not silver_path.exists():
        logger.error(f"Silver trends not found at {silver_path}")
        return

    logger.info("Loading Silver Trends...")
    df = pd.read_parquet(silver_path)
    
    # 1. Identify Top Types
    top_types = df['type_id'].value_counts().head(5).index.tolist()
    logger.info(f"Top 5 Trend Types: {top_types}")
    
    # 2. Pivot to Wide Format (One row per minute? No, aggregate per match first)
    # We want aggregate per match per team
    # e.g. match_id, team_id, type_43_mean, type_43_std
    
    # Filter only top types
    df = df[df['type_id'].isin(top_types)]
    
    # Pivot
    # Group by match, team, type -> Calculate mean/std
    grouped = df.groupby(['match_id', 'team_id', 'type_id'])['value'].agg(['mean', 'std']).unstack()
    
    # Flatten columns
    grouped.columns = [f"trend_{t}_{stat}" for stat, t in grouped.columns]
    grouped = grouped.reset_index()
    
    # Now we have match-level stats.
    # We need to add 'date' to sort for rolling calc.
    # We need to fetch date from player stats or team vectors?
    # Let's load dates from team_vectors_gold (it has match_id) or just fixture_map?
    # Or simpler: load player_features_gold just to get match_id -> date mapping.
    
    matches_path = PROJECT_ROOT / "data" / "processed" / "matches" / "matches_silver.parquet"
    if matches_path.exists():
        dates = pd.read_parquet(matches_path, columns=['match_id', 'date'])
        dates['date'] = pd.to_datetime(dates['date'])
        grouped = grouped.merge(dates, on='match_id', how='left')
    else:
        logger.error("matches_silver.parquet not found. Cannot determine true match order.")
        raise FileNotFoundError("matches_silver.parquet is strictly required for accurate momentum calculation without temporal leakage.")
        
    # --- Phase 9: Playstyle Classifiers ---
    player_stats_path = PROJECT_ROOT / "data" / "processed" / "players" / "player_stats_silver.parquet"
    if player_stats_path.exists():
        logger.info("Integrating Playstyle Classifiers from Player Stats...")
        # Read the newly extracted player stats columns
        try:
            ps_df = pd.read_parquet(player_stats_path, columns=['match_id', 'team_id', 'passes', 'accurate_passes', 'total_duels', 'duels_won'])
            ps_df = ps_df.fillna(0)
            ps_grouped = ps_df.groupby(['match_id', 'team_id']).sum().reset_index()
            
            # Calculate ratios safely using NaN instead of 0.0 for division by zero
            ps_grouped['trend_pass_accuracy'] = np.where(ps_grouped['passes'] > 0, ps_grouped['accurate_passes'] / ps_grouped['passes'], np.nan)
            ps_grouped['trend_possession_proxy'] = ps_grouped['passes']
            ps_grouped['trend_duel_win_rate'] = np.where(ps_grouped['total_duels'] > 0, ps_grouped['duels_won'] / ps_grouped['total_duels'], np.nan)
            
            ps_cols_to_keep = ['match_id', 'team_id', 'trend_pass_accuracy', 'trend_possession_proxy', 'trend_duel_win_rate']
            ps_grouped = ps_grouped[ps_cols_to_keep]
            
            grouped = grouped.merge(ps_grouped, on=['match_id', 'team_id'], how='left')
        except Exception as e:
            logger.error(f"Failed to integrate Playstyle Classifiers: {e}")
            grouped['trend_pass_accuracy'] = np.nan
            grouped['trend_possession_proxy'] = np.nan
            grouped['trend_duel_win_rate'] = np.nan
    else:
        logger.warning("player_stats_silver.parquet not found. Skipping Playstyle Classifiers.")
        grouped['trend_pass_accuracy'] = np.nan
        grouped['trend_possession_proxy'] = np.nan
        grouped['trend_duel_win_rate'] = np.nan
    # -------------------------------------

    grouped = grouped.sort_values(['team_id', 'date'])
    
    # 3. Rolling Features
    cols_to_roll = [c for c in grouped.columns if 'trend_' in c]
    
    logger.info(f"Calculating rolling stats for {len(cols_to_roll)} features...")
    
    for col in cols_to_roll:
        # Fill NaNs with median before rolling to prevent artificial 0.0 outliers
        median_val = grouped[col].median()
        if pd.isna(median_val):
            median_val = 0.0
        grouped[col] = grouped[col].fillna(median_val)
        
        grouped[f'{col}_last5'] = grouped.groupby('team_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
        )
        
    # Select final columns
    final_cols = ['match_id', 'team_id'] + [f'{c}_last5' for c in cols_to_roll]
    out_df = grouped[final_cols].copy()
    out_df = out_df.fillna(0)
    
    # Save
    out_path = PROJECT_ROOT / "data" / "processed" / "teams" / "momentum_features_gold.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    out_df.to_parquet(out_path)
    logger.info(f"Saved {len(out_df)} momentum vectors to {out_path}")
    print(out_df.tail())

if __name__ == "__main__":
    build_momentum()
