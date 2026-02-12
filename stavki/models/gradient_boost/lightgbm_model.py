"""
LightGBM Model for Match Outcome Prediction
=============================================

Gradient boosting model for 1X2 (match winner) prediction.
Features:
- Temporal train/valid split (no data leakage)
- Probability calibration (Isotonic)
- Feature importance tracking
- Per-league support
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import pickle

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import LabelEncoder

from ..base import BaseModel, CalibratedModel, Prediction, Market

logger = logging.getLogger(__name__)


# Default features for 1X2 prediction
DEFAULT_FEATURES = [
    # ELO features
    "HomeEloBefore", "AwayEloBefore", "EloDiff",
    "EloExpHome", "EloExpAway",
    
    # Form features (last 5 matches)
    "Home_GF_L5", "Home_GA_L5", "Home_Pts_L5",
    "Away_GF_L5", "Away_GA_L5", "Away_Pts_L5",
    
    # Overall form (home+away)
    "Home_Overall_GF_L5", "Home_Overall_GA_L5", "Home_Overall_Pts_L5",
    "Away_Overall_GF_L5", "Away_Overall_GA_L5", "Away_Overall_Pts_L5",
    
    # xG features
    "xG_Home_L5", "xGA_Home_L5", "xG_Away_L5", "xGA_Away_L5", "xG_Diff",
    
    # Market signals
    "Sharp_Divergence", "Odds_Volatility", "Market_Consensus",
    
    # H2H
    "H2H_Home_Win_Pct", "H2H_Goals_Avg",
    
    # CLV
    "CLV_Home", "CLV_Draw", "CLV_Away",
]


class LightGBMModel(CalibratedModel):
    """
    LightGBM model for 1X2 prediction with calibration.
    
    Uses gradient boosting with:
    - Multi-class classification (3 classes: H/D/A)
    - Class weights for imbalanced outcomes
    - Early stopping on validation set
    - Isotonic calibration for probability accuracy
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        random_state: int = 42,
        features: Optional[List[str]] = None,
        calibration_method: str = "isotonic",
    ):
        super().__init__(
            name="LightGBM_1X2",
            markets=[Market.MATCH_WINNER],
            calibration_method=calibration_method
        )
        
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.random_state = random_state
        
        self.features = features or DEFAULT_FEATURES
        self.model: Optional[lgb.LGBMClassifier] = None
        self.label_encoder = LabelEncoder()
        self.feature_importance_: Optional[Dict[str, float]] = None
        
        # Calibrators per class
        self.calibrators: Dict[int, IsotonicRegression] = {}
    
    def fit(
        self, 
        data: pd.DataFrame,
        eval_ratio: float = 0.2,
        early_stopping_rounds: int = 50,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train the model using temporal split.
        
        Args:
            data: DataFrame with features and match results
            eval_ratio: Fraction of data for validation (from end)
            early_stopping_rounds: Stop if no improvement
        
        Returns:
            Training metrics
        """
        df = data.copy()
        
        # Ensure date ordering
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        # Create target
        df["target"] = self._create_target(df)
        
        # Get available features
        available_features = [f for f in self.features if f in df.columns]
        if len(available_features) < 5:
            raise ValueError(f"Too few features available. Found: {available_features}")
        
        logger.info(f"Using {len(available_features)} features")
        
        # Temporal split
        n = len(df)
        split_idx = int(n * (1 - eval_ratio))
        
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train_df)}, Eval: {len(eval_df)}")
        
        # Prepare data
        X_train = train_df[available_features].fillna(0)
        y_train = train_df["target"]
        X_eval = eval_df[available_features].fillna(0)
        y_eval = eval_df["target"]
        
        # Encode labels
        y_train_enc = self.label_encoder.fit_transform(y_train)
        y_eval_enc = self.label_encoder.transform(y_eval)
        
        # Class weights (handle imbalance)
        class_counts = np.bincount(y_train_enc)
        class_weights = {i: len(y_train_enc) / (len(class_counts) * count) 
                        for i, count in enumerate(class_counts)}
        
        # Train model
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            feature_fraction=self.feature_fraction,
            bagging_fraction=self.bagging_fraction,
            bagging_freq=self.bagging_freq,
            random_state=self.random_state,
            class_weight=class_weights,
            n_jobs=-1,
            verbose=-1,
        )
        
        self.model.fit(
            X_train, y_train_enc,
            eval_set=[(X_eval, y_eval_enc)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        
        # Feature importance
        self.feature_importance_ = dict(zip(
            available_features,
            self.model.feature_importances_
        ))
        
        # Calibrate on eval set
        self._fit_calibration(X_eval, y_eval_enc)
        
        # Compute metrics
        train_probs = self.model.predict_proba(X_train)
        eval_probs = self.model.predict_proba(X_eval)
        
        train_acc = (train_probs.argmax(axis=1) == y_train_enc).mean()
        eval_acc = (eval_probs.argmax(axis=1) == y_eval_enc).mean()
        
        # Log loss
        eps = 1e-10
        train_ll = -np.mean(np.log(train_probs[np.arange(len(y_train_enc)), y_train_enc] + eps))
        eval_ll = -np.mean(np.log(eval_probs[np.arange(len(y_eval_enc)), y_eval_enc] + eps))
        
        self.is_fitted = True
        self.is_calibrated = True
        self.metadata["n_features"] = len(available_features)
        self.metadata["features"] = available_features
        
        return {
            "train_accuracy": float(train_acc),
            "eval_accuracy": float(eval_acc),
            "train_log_loss": float(train_ll),
            "eval_log_loss": float(eval_ll),
            "n_train": len(train_df),
            "n_eval": len(eval_df),
            "best_iteration": self.model.best_iteration_,
        }
    
    def _create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target variable from match results."""
        def get_result(row):
            if pd.isna(row.get("FTHG")) or pd.isna(row.get("FTAG")):
                return None
            if row["FTHG"] > row["FTAG"]:
                return "H"
            elif row["FTHG"] < row["FTAG"]:
                return "A"
            else:
                return "D"
        
        # Try FTR column first, else compute from goals
        if "FTR" in df.columns:
            return df["FTR"]
        return df.apply(get_result, axis=1)
    
    def _fit_calibration(self, X_eval: pd.DataFrame, y_eval: np.ndarray):
        """Fit isotonic calibration on validation set."""
        probs = self.model.predict_proba(X_eval)
        
        for class_idx in range(probs.shape[1]):
            y_binary = (y_eval == class_idx).astype(int)
            
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(probs[:, class_idx], y_binary)
            self.calibrators[class_idx] = calibrator
 
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate calibrated 1X2 predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get available features
        available_features = self.metadata.get("features", self.features)
        available_features = [f for f in available_features if f in data.columns]
        
        X = data[available_features].fillna(0)
        
        # Raw predictions
        raw_probs = self.model.predict_proba(X)
        
        # Calibrate
        if self.is_calibrated:
            calibrated = np.zeros_like(raw_probs)
            for class_idx in range(raw_probs.shape[1]):
                if class_idx in self.calibrators:
                    calibrated[:, class_idx] = self.calibrators[class_idx].predict(
                        raw_probs[:, class_idx]
                    )
                else:
                    calibrated[:, class_idx] = raw_probs[:, class_idx]
            
            # Renormalize
            row_sums = calibrated.sum(axis=1, keepdims=True)
            calibrated = calibrated / row_sums
            probs = calibrated
        else:
            probs = raw_probs
        
        # Create predictions
        predictions = []
        classes = self.label_encoder.classes_  # ['A', 'D', 'H']
        
        for idx, row in data.iterrows():
            i = data.index.get_loc(idx)
            match_id = row.get("match_id", f"{row.get('HomeTeam', 'home')}_vs_{row.get('AwayTeam', 'away')}_{idx}")
            
            # Map to standard names
            prob_dict = {}
            for j, cls in enumerate(classes):
                if cls == "H":
                    prob_dict["home"] = float(probs[i, j])
                elif cls == "D":
                    prob_dict["draw"] = float(probs[i, j])
                elif cls == "A":
                    prob_dict["away"] = float(probs[i, j])
            
            # Confidence = gap between best and second best
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
        if self.feature_importance_ is None:
            return []
        
        sorted_features = sorted(
            self.feature_importance_.items(),
            key=lambda x: -x[1]
        )
        return sorted_features[:top_n]
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "num_leaves": self.num_leaves,
                "min_child_samples": self.min_child_samples,
                "feature_fraction": self.feature_fraction,
                "bagging_fraction": self.bagging_fraction,
                "bagging_freq": self.bagging_freq,
                "random_state": self.random_state,
            },
            "features": self.features,
            "model": self.model,
            "label_encoder": self.label_encoder,
            "calibrators": self.calibrators,
            "feature_importance": self.feature_importance_,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        params = state["params"]
        self.n_estimators = params["n_estimators"]
        self.learning_rate = params["learning_rate"]
        self.max_depth = params["max_depth"]
        self.num_leaves = params["num_leaves"]
        self.min_child_samples = params["min_child_samples"]
        self.feature_fraction = params["feature_fraction"]
        self.bagging_fraction = params["bagging_fraction"]
        self.bagging_freq = params["bagging_freq"]
        self.random_state = params["random_state"]
        
        self.features = state["features"]
        self.model = state["model"]
        self.label_encoder = state["label_encoder"]
        self.calibrators = state["calibrators"]
        self.feature_importance_ = state["feature_importance"]
