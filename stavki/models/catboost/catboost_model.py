"""
CatBoost Model for Match Outcome Prediction
============================================

CatBoost (Categorical Boosting) - Preferred over LightGBM for football betting:

**Why CatBoost > LightGBM:**
1. **Native categorical support** - No encoding needed, handles team names, leagues directly
2. **Ordered boosting** - Prevents target leakage (critical for temporal data)
3. **Less overfitting** - Built-in regularization works better on small datasets
4. **Better calibration** - Probabilities are more accurate out-of-box
5. **Production proven** - Used by Yandex for high-stakes predictions

**Research (2024):**
- Kaggle comparison: CatBoost 96.2% vs LightGBM 95.9% accuracy
- Football prediction: CatBoost excels with league/team categoricals
- Calibration: CatBoost predictions need less post-hoc adjustment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    CatBoostClassifier = None
    Pool = None

from sklearn.isotonic import IsotonicRegression

from ..base import BaseModel, CalibratedModel, Prediction, Market

logger = logging.getLogger(__name__)


# Features for CatBoost (includes categoricals)
CATBOOST_FEATURES = [
    # Numeric features
    "HomeEloBefore", "AwayEloBefore", "EloDiff",
    "EloExpHome", "EloExpAway",
    "Home_GF_L5", "Home_GA_L5", "Home_Pts_L5",
    "Away_GF_L5", "Away_GA_L5", "Away_Pts_L5",
    "Home_Overall_GF_L5", "Home_Overall_GA_L5",
    "Away_Overall_GF_L5", "Away_Overall_GA_L5",
    "xG_Home_L5", "xGA_Home_L5", "xG_Away_L5", "xGA_Away_L5",
    "Sharp_Divergence", "Odds_Volatility",
    "H2H_Home_Win_Pct", "H2H_Goals_Avg",
]

# Categorical features (CatBoost handles natively)
CATEGORICAL_FEATURES = ["League", "HomeTeam", "AwayTeam"]


class CatBoostModel(CalibratedModel):
    """
    CatBoost classifier for 1X2 prediction.
    
    Advantages over LightGBM:
    - Native categorical feature handling
    - Ordered boosting prevents data leakage
    - Better calibrated probabilities
    - Robust to overfitting on small datasets
    """
    
    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_strength: float = 1.0,
        bagging_temperature: float = 1.0,
        random_seed: int = 42,
        features: Optional[List[str]] = None,
        cat_features: Optional[List[str]] = None,
        use_gpu: bool = False,
    ):
        super().__init__(
            name="CatBoost_1X2",
            markets=[Market.MATCH_WINNER],
            calibration_method="isotonic"
        )
        
        if not HAS_CATBOOST:
            raise ImportError("catboost required. Install: pip install catboost")
        
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.random_seed = random_seed
        self.use_gpu = use_gpu
        
        self.features = features or CATBOOST_FEATURES
        self.cat_features = cat_features or CATEGORICAL_FEATURES
        
        self.model: Optional[CatBoostClassifier] = None
        self.calibrators: Dict[int, IsotonicRegression] = {}
        self.feature_importance_: Optional[Dict[str, float]] = None
    
    def fit(
        self,
        data: pd.DataFrame,
        eval_ratio: float = 0.15,
        early_stopping_rounds: int = 50,
        verbose: int = 0,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train CatBoost with ordered boosting.
        
        Uses temporal split (critical for betting - no data leakage).
        """
        df = data.copy()
        
        # Sort by date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        # Create target
        df["target"] = self._create_target(df)
        df = df.dropna(subset=["target"])
        
        # Get available features (numeric + categorical)
        available_numeric = [f for f in self.features if f in df.columns]
        available_cat = [f for f in self.cat_features if f in df.columns]
        all_features = available_numeric + available_cat
        
        if len(all_features) < 3:
            # Fallback
            available_numeric = [c for c in df.columns 
                                if df[c].dtype in [np.float64, np.int64]
                                and c not in ["FTHG", "FTAG", "target"]][:20]
            all_features = available_numeric
            available_cat = []
        
        logger.info(f"Using {len(available_numeric)} numeric + {len(available_cat)} categorical features")
        
        # Temporal split
        n = len(df)
        split_idx = int(n * (1 - eval_ratio))
        
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]
        
        # Prepare data
        X_train = train_df[all_features].copy()
        y_train = train_df["target"].values.astype(int)
        X_eval = eval_df[all_features].copy()
        y_eval = eval_df["target"].values.astype(int)
        
        # Handle categoricals
        for cat_col in available_cat:
            X_train[cat_col] = X_train[cat_col].fillna("Unknown").astype(str)
            X_eval[cat_col] = X_eval[cat_col].fillna("Unknown").astype(str)
        
        # Fill numeric NaNs
        for num_col in available_numeric:
            X_train[num_col] = X_train[num_col].fillna(0)
            X_eval[num_col] = X_eval[num_col].fillna(0)
        
        # Cat feature indices
        cat_feature_indices = [all_features.index(c) for c in available_cat if c in all_features]
        
        # Create pools
        train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
        eval_pool = Pool(X_eval, y_eval, cat_features=cat_feature_indices)
        
        # Class weights
        class_counts = np.bincount(y_train, minlength=3)
        class_weights = [len(y_train) / (3 * max(c, 1)) for c in class_counts]
        
        # Train
        self.model = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature,
            random_seed=self.random_seed,
            loss_function="MultiClass",
            class_weights=class_weights,
            task_type="GPU" if self.use_gpu else "CPU",
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )
        
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            use_best_model=True,
        )
        
        # Feature importance
        importance = self.model.get_feature_importance()
        self.feature_importance_ = dict(zip(all_features, importance))
        
        # Calibrate
        eval_probs = self.model.predict_proba(X_eval)
        for class_idx in range(3):
            y_binary = (y_eval == class_idx).astype(int)
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(eval_probs[:, class_idx], y_binary)
            self.calibrators[class_idx] = calibrator
        
        # Metrics
        train_probs = self.model.predict_proba(X_train)
        train_preds = train_probs.argmax(axis=1)
        eval_preds = eval_probs.argmax(axis=1)
        
        train_acc = (train_preds == y_train).mean()
        eval_acc = (eval_preds == y_eval).mean()
        
        # Log loss
        eps = 1e-10
        train_ll = -np.mean(np.log(train_probs[np.arange(len(y_train)), y_train] + eps))
        eval_ll = -np.mean(np.log(eval_probs[np.arange(len(y_eval)), y_eval] + eps))
        
        self.is_fitted = True
        self.is_calibrated = True
        self.metadata["features"] = all_features
        self.metadata["cat_features"] = available_cat
        self.metadata["cat_indices"] = cat_feature_indices
        
        return {
            "train_accuracy": float(train_acc),
            "eval_accuracy": float(eval_acc),
            "train_log_loss": float(train_ll),
            "eval_log_loss": float(eval_ll),
            "best_iteration": self.model.best_iteration_,
            "n_train": len(train_df),
            "n_eval": len(eval_df),
        }
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target: 0=Home, 1=Draw, 2=Away."""
        if "FTR" in df.columns:
            return df["FTR"].map({"H": 0, "D": 1, "A": 2})
        
        def get_result(row):
            if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                return None
            if row["FTHG"] > row["FTAG"]:
                return 0  # Home
            elif row["FTHG"] < row["FTAG"]:
                return 2  # Away
            return 1  # Draw
        
        return df.apply(get_result, axis=1)
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate calibrated 1X2 predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        all_features = self.metadata.get("features", [])
        cat_features = self.metadata.get("cat_features", [])
        
        X = data[all_features].copy() if all(f in data.columns for f in all_features) else data.copy()
        
        # Handle missing features
        available = [f for f in all_features if f in data.columns]
        X = data[available].copy()
        
        # Process categoricals
        for cat_col in cat_features:
            if cat_col in X.columns:
                X[cat_col] = X[cat_col].fillna("Unknown").astype(str)
        
        # Fill numeric
        for col in X.columns:
            if col not in cat_features:
                X[col] = X[col].fillna(0)
        
        # Predict
        raw_probs = self.model.predict_proba(X)
        
        # Calibrate
        calibrated = np.zeros_like(raw_probs)
        for class_idx in range(3):
            if class_idx in self.calibrators:
                calibrated[:, class_idx] = self.calibrators[class_idx].predict(raw_probs[:, class_idx])
            else:
                calibrated[:, class_idx] = raw_probs[:, class_idx]
        
        # Normalize
        row_sums = calibrated.sum(axis=1, keepdims=True)
        calibrated = calibrated / row_sums
        
        predictions = []
        for idx, row in data.iterrows():
            i = data.index.get_loc(idx)
            match_id = row.get("match_id", f"{row.get('HomeTeam', 'home')}_vs_{row.get('AwayTeam', 'away')}_{idx}")
            
            # Note: CatBoost uses 0=Home, 1=Draw, 2=Away
            prob_dict = {
                "home": float(calibrated[i, 0]),
                "draw": float(calibrated[i, 1]),
                "away": float(calibrated[i, 2]),
            }
            
            sorted_probs = sorted(prob_dict.values(), reverse=True)
            confidence = sorted_probs[0] - sorted_probs[1]
            
            predictions.append(Prediction(
                match_id=match_id,
                market=Market.MATCH_WINNER,
                probabilities=prob_dict,
                confidence=confidence,
                model_name=self.name,
            ))
        
        return predictions
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """Return top N features by importance."""
        if not self.feature_importance_:
            return []
        
        sorted_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: -x[1]
        )
        return sorted_features[:top_n]
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "params": {
                "iterations": self.iterations,
                "learning_rate": self.learning_rate,
                "depth": self.depth,
                "l2_leaf_reg": self.l2_leaf_reg,
                "random_strength": self.random_strength,
                "bagging_temperature": self.bagging_temperature,
                "random_seed": self.random_seed,
                "use_gpu": self.use_gpu,
            },
            "features": self.features,
            "cat_features": self.cat_features,
            "model": self.model,
            "calibrators": self.calibrators,
            "feature_importance": self.feature_importance_,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        params = state["params"]
        for key, value in params.items():
            setattr(self, key, value)
        
        self.features = state["features"]
        self.cat_features = state["cat_features"]
        self.model = state["model"]
        self.calibrators = state["calibrators"]
        self.feature_importance_ = state["feature_importance"]
