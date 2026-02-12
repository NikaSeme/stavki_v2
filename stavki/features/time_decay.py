"""
Time Decay Weighting for Model Training.

Implements season-based exponential decay to weight recent data more heavily.
Old data is less valuable due to:
- Squad changes (transfers)
- Tactical evolution
- Market efficiency improvements
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

# Default decay factor (30% weight loss per season)
DEFAULT_DECAY_FACTOR = 0.7


def get_season_from_date(date: str) -> str:
    """
    Extract season from date string.
    
    Football season runs Aug-May, so:
    - Jan-Jul 2024 = 2023/24 season
    - Aug-Dec 2024 = 2024/25 season
    
    Args:
        date: Date string in YYYY-MM-DD format
        
    Returns:
        Season string like '2024/25'
    """
    try:
        if isinstance(date, str):
            dt = datetime.strptime(date[:10], "%Y-%m-%d")
        else:
            dt = pd.to_datetime(date)
        
        year = dt.year
        month = dt.month
        
        # Before August = previous season
        if month < 8:
            start_year = year - 1
        else:
            start_year = year
        
        return f"{start_year}/{str(start_year + 1)[-2:]}"
    
    except Exception:
        return "unknown"


def get_current_season() -> str:
    """Get the current football season."""
    return get_season_from_date(datetime.now().strftime("%Y-%m-%d"))


def season_to_number(season: str) -> int:
    """
    Convert season string to comparable number.
    
    Args:
        season: Season string like '2024/25'
        
    Returns:
        Integer year (start year of season)
    """
    try:
        return int(season.split('/')[0])
    except Exception:
        return 2000  # Default old


def calculate_seasons_ago(date: str, reference_season: Optional[str] = None) -> int:
    """
    Calculate how many seasons ago a date was.
    
    Args:
        date: Date string
        reference_season: Reference season (default: current)
        
    Returns:
        Number of seasons ago (0 = current season)
    """
    if reference_season is None:
        reference_season = get_current_season()
    
    match_season = get_season_from_date(date)
    
    ref_year = season_to_number(reference_season)
    match_year = season_to_number(match_season)
    
    return max(0, ref_year - match_year)


def calculate_season_weight(
    date: str,
    decay_factor: float = DEFAULT_DECAY_FACTOR,
    reference_season: Optional[str] = None,
    min_weight: float = 0.05
) -> float:
    """
    Calculate sample weight based on season decay.
    
    Args:
        date: Match date
        decay_factor: Decay per season (0.7 = 30% loss per season)
        reference_season: Reference for "current" (default: now)
        min_weight: Minimum weight to avoid zeros
        
    Returns:
        Weight in range [min_weight, 1.0]
    """
    seasons_ago = calculate_seasons_ago(date, reference_season)
    weight = decay_factor ** seasons_ago
    return max(min_weight, weight)


def add_sample_weights(
    df: pd.DataFrame,
    date_column: str = 'Date',
    decay_factor: float = DEFAULT_DECAY_FACTOR,
    reference_season: Optional[str] = None,
    weight_column: str = 'sample_weight'
) -> pd.DataFrame:
    """
    Add sample weight column to DataFrame.
    
    Args:
        df: DataFrame with matches
        date_column: Name of date column
        decay_factor: Decay per season
        reference_season: Reference season
        weight_column: Name for weight column
        
    Returns:
        DataFrame with added weight column
    """
    df = df.copy()
    
    df[weight_column] = df[date_column].apply(
        lambda d: calculate_season_weight(d, decay_factor, reference_season)
    )
    
    # Add season column for reference
    df['season'] = df[date_column].apply(get_season_from_date)
    
    return df


def get_weight_summary(df: pd.DataFrame, weight_column: str = 'sample_weight') -> dict:
    """
    Get summary of weight distribution by season.
    
    Returns:
        Dict with season weights and counts
    """
    if weight_column not in df.columns:
        return {"error": "No weight column found"}
    
    summary = df.groupby('season').agg({
        weight_column: ['mean', 'count']
    }).round(3)
    
    summary.columns = ['avg_weight', 'matches']
    
    return summary.to_dict('index')


def optimize_decay_factor(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = 'target',
    date_col: str = 'Date',
    decay_factors: list = None,
    cv_folds: int = 3
) -> Tuple[float, dict]:
    """
    Find optimal decay factor through cross-validation.
    
    Args:
        df: Training data
        feature_cols: Feature column names
        target_col: Target column name
        date_col: Date column name
        decay_factors: List of factors to test
        cv_folds: Number of CV folds
        
    Returns:
        Tuple of (best_factor, results_dict)
    """
    from catboost import CatBoostClassifier
    from sklearn.model_selection import cross_val_score
    
    if decay_factors is None:
        decay_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    results = {}
    
    for factor in decay_factors:
        # Add weights
        df_weighted = add_sample_weights(df, date_col, factor)
        
        X = df_weighted[feature_cols].fillna(0)
        y = df_weighted[target_col]
        weights = df_weighted['sample_weight'].values
        
        # Train with weights
        model = CatBoostClassifier(iterations=100, verbose=0)
        
        # Use weighted training in CV
        scores = []
        fold_size = len(df) // cv_folds
        
        for i in range(cv_folds):
            val_start = i * fold_size
            val_end = val_start + fold_size
            
            train_idx = list(range(0, val_start)) + list(range(val_end, len(df)))
            val_idx = list(range(val_start, val_end))
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            w_train = weights[train_idx]
            
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            model.fit(X_train, y_train, sample_weight=w_train, verbose=0)
            score = model.score(X_val, y_val)
            scores.append(score)
        
        results[factor] = {
            'accuracy': np.mean(scores),
            'std': np.std(scores)
        }
    
    # Find best
    best_factor = max(results.keys(), key=lambda k: results[k]['accuracy'])
    
    return best_factor, results


# Convenience: Pre-calculated weights for common seasons
SEASON_WEIGHTS = {
    "2024/25": 1.000,
    "2023/24": 0.700,
    "2022/23": 0.490,
    "2021/22": 0.343,
    "2020/21": 0.240,
    "2019/20": 0.168,
    "2018/19": 0.118,
    "2017/18": 0.082,
    "2016/17": 0.058,
    "2015/16": 0.040,
    "2014/15": 0.028,
}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/Users/macuser/Documents/something/stavki_v2')
    
    from stavki.config import DATA_DIR
    
    print("="*60)
    print("📊 SEASON DECAY WEIGHTING DEMO")
    print("="*60)
    
    # Load data
    df = pd.read_csv(DATA_DIR / 'features_full.csv', low_memory=False)
    
    print(f"\nLoaded {len(df)} matches")
    
    # Add weights
    df = add_sample_weights(df, decay_factor=0.7)
    
    print("\nWeight distribution by season:")
    print("-" * 40)
    
    summary = df.groupby('season').agg({
        'sample_weight': 'mean',
        'HomeTeam': 'count'
    }).rename(columns={'HomeTeam': 'matches'})
    
    summary = summary.sort_index(ascending=False)
    
    for season, row in summary.iterrows():
        bar = "█" * int(row['sample_weight'] * 20)
        print(f"  {season}: {row['sample_weight']:.3f} {bar} ({int(row['matches'])} matches)")
    
    print("\n" + "="*60)
    print(f"Total weighted samples: {df['sample_weight'].sum():.0f}")
    print(f"Effective sample size: {df['sample_weight'].sum():.0f} / {len(df)} = {df['sample_weight'].sum()/len(df):.1%}")
