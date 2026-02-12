"""
BTTS (Both Teams To Score) Classifier
======================================

LightGBM binary classifier for BTTS market.
Features specialized for goal-scoring patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    lgb = None

from sklearn.isotonic import IsotonicRegression

from ..base import BaseModel, CalibratedModel, Prediction, Market

logger = logging.getLogger(__name__)


# BTTS-specific features
BTTS_FEATURES = [
    # Scoring patterns
    "Home_GF_L5", "Home_GA_L5",
    "Away_GF_L5", "Away_GA_L5",
    "Home_Overall_GF_L5", "Away_Overall_GF_L5",
    
    # Clean sheets / scoring probability
    "Home_Clean_Sheet_Pct", "Away_Clean_Sheet_Pct",
    "Home_Scored_Pct", "Away_Scored_Pct",
    "Home_Failed_Score_Pct", "Away_Failed_Score_Pct",
    
    # xG features
    "xG_Home_L5", "xGA_Home_L5", "xG_Away_L5", "xGA_Away_L5",
    
    # Defense strength
    "Home_Defense_Rating", "Away_Defense_Rating",
    "Home_Attack_Rating", "Away_Attack_Rating",
    
    # ELO components
    "HomeEloBefore", "AwayEloBefore",
    
    # H2H BTTS history
    "H2H_BTTS_Pct", "H2H_Goals_Per_Game",
    
    # League BTTS rate
    "League_BTTS_Rate",
]


class BTTSModel(CalibratedModel):
    """
    BTTS binary classifier using LightGBM.
    
    Optimized for predicting whether both teams will score.
    """
    
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        num_leaves: int = 24,
        min_child_samples: int = 25,
        random_state: int = 42,
        features: Optional[List[str]] = None,
    ):
        super().__init__(
            name="LightGBM_BTTS",
            markets=[Market.BTTS],
            calibration_method="isotonic"
        )
        
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm required")
        
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.random_state = random_state
        
        self.features = features or BTTS_FEATURES
        self.model: Optional[lgb.LGBMClassifier] = None
        self.calibrator: Optional[IsotonicRegression] = None
    
    def fit(
        self, 
        data: pd.DataFrame,
        eval_ratio: float = 0.2,
        early_stopping_rounds: int = 30,
        **kwargs
    ) -> Dict[str, float]:
        """Train BTTS classifier."""
        df = data.copy()
        
        # Sort by date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        # Create BTTS target: 1 if both teams scored
        df["btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        # Get features
        available = [f for f in self.features if f in df.columns]
        
        # Add computed features if missing
        df = self._add_computed_features(df)
        available = [f for f in self.features if f in df.columns]
        
        if len(available) < 3:
            # Fallback to basic features
            available = [c for c in df.columns if df[c].dtype in [np.float64, np.int64]]
            available = [c for c in available if c not in ["FTHG", "FTAG", "btts"]][:20]
        
        logger.info(f"BTTS model using {len(available)} features")
        
        # Split
        n = len(df)
        split_idx = int(n * (1 - eval_ratio))
        
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]
        
        X_train = train_df[available].fillna(0)
        y_train = train_df["btts"]
        X_eval = eval_df[available].fillna(0)
        y_eval = eval_df["btts"]
        
        # Class balance
        pos_rate = y_train.mean()
        scale = (1 - pos_rate) / max(pos_rate, 0.01)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            scale_pos_weight=scale,
            random_state=self.random_state,
            verbose=-1,
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )
        
        # Calibrate
        eval_probs = self.model.predict_proba(X_eval)[:, 1]
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(eval_probs, y_eval)
        
        # Metrics
        train_probs = self.model.predict_proba(X_train)[:, 1]
        train_preds = (train_probs > 0.5).astype(int)
        eval_preds = (eval_probs > 0.5).astype(int)
        
        train_acc = (train_preds == y_train).mean()
        eval_acc = (eval_preds == y_eval).mean()
        
        self.is_fitted = True
        self.is_calibrated = True
        self.metadata["features"] = available
        
        return {
            "train_accuracy": float(train_acc),
            "eval_accuracy": float(eval_acc),
            "btts_rate_train": float(y_train.mean()),
            "btts_rate_eval": float(y_eval.mean()),
        }
    
    def _add_computed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add BTTS-specific computed features."""
        # Clean sheet percentage (if we have the data)
        if "Home_GF_L5" in df.columns and "Home_GA_L5" in df.columns:
            # Approximate clean sheet % from average goals against
            df["Home_Clean_Sheet_Pct"] = np.clip(1 - df["Home_GA_L5"] / 1.5, 0, 1)
            df["Away_Clean_Sheet_Pct"] = np.clip(1 - df["Away_GA_L5"] / 1.5, 0, 1)
            
            # Scoring probability
            df["Home_Scored_Pct"] = np.clip(df["Home_GF_L5"] / 2, 0, 1)
            df["Away_Scored_Pct"] = np.clip(df["Away_GF_L5"] / 2, 0, 1)
        
        return df
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate BTTS predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        features = self.metadata.get("features", [])
        available = [f for f in features if f in data.columns]
        
        X = data[available].fillna(0)
        raw_probs = self.model.predict_proba(X)[:, 1]
        
        # Calibrate
        if self.calibrator:
            probs = self.calibrator.predict(raw_probs)
        else:
            probs = raw_probs
        
        predictions = []
        for idx, row in data.iterrows():
            i = data.index.get_loc(idx)
            match_id = row.get("match_id", f"{row.get('HomeTeam', 'home')}_vs_{row.get('AwayTeam', 'away')}")
            
            predictions.append(Prediction(
                match_id=match_id,
                market=Market.BTTS,
                probabilities={
                    "yes": float(probs[i]),
                    "no": float(1 - probs[i]),
                },
                confidence=abs(probs[i] - 0.5) * 2,
                model_name=self.name,
            ))
        
        return predictions
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "params": {
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
                "num_leaves": self.num_leaves,
                "min_child_samples": self.min_child_samples,
                "random_state": self.random_state,
            },
            "features": self.features,
            "model": self.model,
            "calibrator": self.calibrator,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        params = state["params"]
        self.n_estimators = params["n_estimators"]
        self.learning_rate = params["learning_rate"]
        self.max_depth = params["max_depth"]
        self.num_leaves = params["num_leaves"]
        self.min_child_samples = params["min_child_samples"]
        self.random_state = params["random_state"]
        
        self.features = state["features"]
        self.model = state["model"]
        self.calibrator = state["calibrator"]
