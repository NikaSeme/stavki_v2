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


# Default features for 1X2 prediction — must match real column names in features_full.csv
DEFAULT_FEATURES = [
    # ELO features
    "elo_home", "elo_away", "elo_diff",
    
    # Form features (last 5 matches)
    "form_home_pts", "form_away_pts", "form_diff",
    "form_home_gf", "form_away_gf",
    "gf_diff", "ga_diff",
    
    # Odds features (Bet365 as primary source)
    "B365H", "B365D", "B365A",
    "imp_home_norm", "imp_draw_norm", "imp_away_norm",
    "margin",
    
    # Tier 1: Synthetic xG
    "synth_xg_home", "synth_xg_away", "synth_xg_diff",
    
    # Tier 1: Player ratings
    "avg_rating_home", "avg_rating_away", "rating_delta",
    "key_players_home", "key_players_away",
    "xi_experience_home", "xi_experience_away",
    
    # Tier 1: Referee profile
    "ref_goals_per_game", "ref_cards_per_game_t1",
    "ref_over25_rate", "ref_strictness_t1",
    "ref_experience", "ref_goals_zscore",
    
    # Phase 3: Formation matchup
    "formation_score_home", "formation_score_away",
    "formation_mismatch", "formation_is_known",
    "matchup_home_wr", "matchup_sample_size",
    
    # Phase 3: Rolling match stats
    "rolling_fouls_home", "rolling_fouls_away",
    "rolling_yellows_home", "rolling_yellows_away",
    "rolling_corners_home", "rolling_corners_away",
    "rolling_possession_home", "rolling_possession_away",
    
    # Phase 3: Referee target encoding
    "ref_encoded_goals", "ref_encoded_cards",
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
        
        self.features = features # If None, will auto-detect
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
        if "target" not in df.columns:
             df["target"] = self._create_target(df)
        
        # Auto-detect features if not specified
        if self.features is None:
             # Exclude non-feature columns and match stats (LEAKAGE)
             exclude = {
                 "target", "Date", "FTHG", "FTAG", "FTR", "match_id", "id", 
                 "HomeTeam", "AwayTeam", "home_team", "away_team", "league", "season",
                 "Referee", "Time", "Div", "Season",
                 # Match stats (Leakage)
                 "HTHG", "HTAG", "HTR", 
                 "HS", "AS", "HST", "AST", 
                 "HC", "AC", "HF", "AF",
                 "HY", "AY", "HR", "AR",
             }
             self.features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
             
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
        """Generate calibrated 1X2 predictions using vectorized operations."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get expected features from Booster if available (Source of Truth)
        try:
             # LightGBM sklearn API stores booster in booster_
             if hasattr(self.model, "booster_"):
                 expected_features = self.model.booster_.feature_name()
             else:
                 # Fallback
                 expected_features = self.metadata.get("features", self.features)
        except Exception as e:
             logger.warning(f"Failed to get features from booster: {e}")
             expected_features = self.metadata.get("features", self.features)

        if not expected_features:
             # Should not happen if model is fitted
             logger.warning("No expected features found in metadata or booster")
             expected_features = []

        # Check for missing features
        missing = [f for f in expected_features if f not in data.columns]
        if missing:
             logger.warning(f"LightGBM missing {len(missing)} features: {missing[:5]}... (Total expected: {len(expected_features)})")
             # logger.debug(f"Missing list: {missing}")
        
        # Strict selection: Use exact features model expects
        # Note: LightGBM is robust to missing cols if using Pandas, but here we construct X manually.
        # We must ensure X has exactly 'expected_features' in order.
        
        # Create X with all expected features, filling missing with 0
        # This is safer than just selecting what's available
        X = data.reindex(columns=expected_features, fill_value=0)
        
        # Raw predictions (Vectorized)
        raw_probs = self.model.predict_proba(X)
        
        # Calibrate (Vectorized)
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
            # Avoid div by zero
            calibrated = np.divide(calibrated, row_sums, where=row_sums!=0)
            probs = calibrated
        else:
            probs = raw_probs
        
        # Vectorized Match ID Generation
        from stavki.utils import generate_match_id
        # Safe vectorized generation
        temp_id = data.copy()
        # Use apply for robustness with existing util
        match_ids = temp_id.apply(
            lambda x: x.get("match_id", generate_match_id(
                x.get("HomeTeam", "home"), 
                x.get("AwayTeam", "away"), 
                x.get("Date")
            )), 
            axis=1
        ).values
        
        # Vectorized Confidence Calculation
        # Sort probabilities for each row
        sorted_probs = np.sort(probs, axis=1)
        # Confidence = best - second_best (last - second_last)
        if probs.shape[1] > 1:
            confidences = sorted_probs[:, -1] - sorted_probs[:, -2]
        else:
            confidences = sorted_probs[:, -1]
            
        # Create Predictions
        predictions = []
        classes = self.label_encoder.classes_  # ['A', 'D', 'H'] (Check alphabetical order usually)
        
        # Optimization: indices for H, D, A
        # LabelEncoder sorts classes alphabetically: 'A', 'D', 'H' typically?
        # Or 'A', 'D', 'H' -> Away, Draw, Home
        # Let's verify mapping dynamically
        map_indices = {}
        for idx, cls in enumerate(classes):
            if cls == "H": map_indices["home"] = idx
            elif cls == "D": map_indices["draw"] = idx
            elif cls == "A": map_indices["away"] = idx
            
        # Fast iteration
        for i in range(len(data)):
            p_vec = probs[i]
            
            prob_dict = {
                k: float(p_vec[v]) for k, v in map_indices.items()
            }
            
            predictions.append(Prediction(
                match_id=match_ids[i],
                market=Market.MATCH_WINNER,
                probabilities=prob_dict,
                confidence=float(confidences[i]),
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
