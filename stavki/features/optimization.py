"""
Feature optimization using Recursive Feature Elimination.

Finds optimal number of features for each model by cross-validated RFE.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from catboost import CatBoostClassifier
import logging

logger = logging.getLogger(__name__)


class FeatureOptimizer:
    """
    Optimize feature selection using RFE and cross-validation.
    
    Usage:
        optimizer = FeatureOptimizer(n_features_range=(5, 15))
        best_features = optimizer.optimize(X, y, feature_names)
    """
    
    def __init__(
        self,
        n_features_range: Tuple[int, int] = (5, 15),
        cv_folds: int = 5,
        random_state: int = 42
    ):
        self.n_features_range = n_features_range
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results_: Dict = {}
        self.best_features_: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
    
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        use_time_series_cv: bool = True
    ) -> List[str]:
        """
        Find optimal features using RFE with cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            feature_names: Optional list of feature names
            use_time_series_cv: Use time-based CV (recommended for backtesting)
            
        Returns:
            List of optimal feature names
        """
        if feature_names is None:
            feature_names = list(X.columns)
        
        X_arr = X[feature_names].fillna(0).values
        y_arr = y.values
        
        # Base estimator
        estimator = CatBoostClassifier(
            iterations=100,
            learning_rate=0.05,
            depth=4,
            verbose=0,
            random_seed=self.random_state
        )
        
        # Cross-validation strategy
        if use_time_series_cv:
            cv = TimeSeriesSplit(n_splits=self.cv_folds)
        else:
            cv = self.cv_folds
        
        logger.info(f"Starting RFE optimization on {len(feature_names)} features")
        
        # Test different numbers of features
        results = {}
        min_n, max_n = self.n_features_range
        
        for n_features in range(min_n, min(max_n + 1, len(feature_names) + 1)):
            rfe = RFE(estimator, n_features_to_select=n_features, step=1)
            
            # Cross-validated score
            try:
                scores = cross_val_score(
                    rfe, X_arr, y_arr, 
                    cv=cv, scoring='accuracy', n_jobs=-1
                )
                mean_score = scores.mean()
                std_score = scores.std()
                
                # Get selected features
                rfe.fit(X_arr, y_arr)
                selected = [f for f, s in zip(feature_names, rfe.support_) if s]
                
                results[n_features] = {
                    'score': mean_score,
                    'std': std_score,
                    'features': selected,
                    'ranking': dict(zip(feature_names, rfe.ranking_))
                }
                
                logger.info(f"  n={n_features}: accuracy={mean_score:.4f} ± {std_score:.4f}")
                
            except Exception as e:
                logger.warning(f"  n={n_features}: failed - {e}")
        
        self.results_ = results
        
        # Find optimal number of features
        if results:
            # Best = highest score, tiebreaker = fewer features
            best_n = max(results.keys(), key=lambda k: (results[k]['score'], -k))
            self.best_features_ = results[best_n]['features']
            
            # Calculate feature importances across all tests
            importance_counts = {}
            for n, data in results.items():
                for feature, rank in data['ranking'].items():
                    if feature not in importance_counts:
                        importance_counts[feature] = []
                    importance_counts[feature].append(1 / rank)  # Lower rank = more important
            
            self.feature_importances_ = {
                f: np.mean(scores) for f, scores in importance_counts.items()
            }
            
            logger.info(f"Optimal features ({best_n}): {self.best_features_}")
            return self.best_features_
        
        return feature_names  # Return all if optimization failed
    
    def get_summary(self) -> str:
        """Get human-readable summary of optimization results."""
        if not self.results_:
            return "No optimization results available."
        
        lines = ["Feature Optimization Results", "=" * 40]
        
        for n, data in sorted(self.results_.items()):
            lines.append(f"n={n}: {data['score']:.4f} ± {data['std']:.4f}")
        
        lines.append("")
        lines.append(f"Best features ({len(self.best_features_)}):")
        for f in self.best_features_:
            imp = self.feature_importances_.get(f, 0)
            lines.append(f"  - {f}: {imp:.4f}")
        
        return "\n".join(lines)


def quick_rfe(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'target',
    n_features: int = 7
) -> List[str]:
    """
    Quick RFE without full optimization - for single run.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        n_features: Number of features to select
        
    Returns:
        List of selected feature names
    """
    X = df[feature_cols].fillna(0)
    y = df[target_col]
    
    estimator = CatBoostClassifier(iterations=50, verbose=0)
    rfe = RFE(estimator, n_features_to_select=n_features)
    rfe.fit(X, y)
    
    selected = [f for f, s in zip(feature_cols, rfe.support_) if s]
    return selected


if __name__ == "__main__":
    # Demo usage
    import sys
    sys.path.insert(0, '/Users/macuser/Documents/something/stavki_v2')
    
    from stavki.config import DATA_DIR
    from sklearn.preprocessing import LabelEncoder
    
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / 'features_full.csv', low_memory=False)
    df = df[df['League'] != 'championship']
    
    # Features
    feature_cols = [
        'elo_diff', 'form_diff', 'gf_diff', 'ga_diff',
        'imp_home_norm', 'imp_draw_norm', 'imp_away_norm',
        'B365H', 'B365D', 'B365A', 'elo_home', 'elo_away',
        'form_home_pts', 'form_away_pts'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    # Target
    le = LabelEncoder()
    df['target'] = le.fit_transform(df['FTR'])
    
    # Use subset for speed
    df_sample = df.tail(5000)
    
    print(f"\nOptimizing {len(feature_cols)} features on {len(df_sample)} samples...")
    optimizer = FeatureOptimizer(n_features_range=(5, 12), cv_folds=3)
    best = optimizer.optimize(df_sample[feature_cols], df_sample['target'], feature_cols)
    
    print()
    print(optimizer.get_summary())
