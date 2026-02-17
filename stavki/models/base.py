"""
Base Model Interface for STAVKI v2
===================================
All prediction models must implement this interface.
Supports multiple markets (1X2, O/U, BTTS, AH).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
import logging

logger = logging.getLogger(__name__)


class Market(Enum):
    """Supported betting markets."""
    MATCH_WINNER = "1x2"        # Home/Draw/Away
    OVER_UNDER = "over_under"   # Over/Under X.5 goals
    BTTS = "btts"               # Both Teams To Score
    ASIAN_HANDICAP = "asian_handicap"
    DOUBLE_CHANCE = "double_chance"
    CORRECT_SCORE = "correct_score"


@dataclass
class Prediction:
    """Standardized prediction output for any market."""
    match_id: str
    market: Market
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Probabilities (market-specific)
    probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Confidence metrics
    confidence: float = 0.0
    model_name: str = ""
    
    # For debugging/analysis
    features_used: Optional[Dict[str, float]] = None
    
    def get_best_outcome(self) -> Tuple[str, float]:
        """Return highest probability outcome."""
        if not self.probabilities:
            return ("", 0.0)
        best = max(self.probabilities.items(), key=lambda x: x[1])
        return best
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "market": self.market.value,
            "timestamp": self.timestamp.isoformat(),
            "probabilities": self.probabilities,
            "confidence": self.confidence,
            "model_name": self.model_name,
        }


@dataclass
class MatchPredictions:
    """All predictions for a single match across markets."""
    match_id: str
    predictions: Dict[Market, Prediction] = field(default_factory=dict)
    
    # Ensemble-level metadata
    ensemble_weights: Dict[str, float] = field(default_factory=dict)
    disagreement_score: float = 0.0
    
    def get(self, market: Market) -> Optional[Prediction]:
        return self.predictions.get(market)
    
    def add(self, prediction: Prediction):
        self.predictions[prediction.market] = prediction


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    
    Each model must:
    1. Implement fit() - train on historical data
    2. Implement predict() - generate predictions
    3. Support save/load for persistence
    4. Declare supported markets
    """
    
    def __init__(self, name: str, markets: List[Market]):
        self.name = name
        self.markets = markets
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "version": "2.0.0",
        }
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train the model on historical match data.
        
        Args:
            data: DataFrame with match results & features
                  Required columns: HomeTeam, AwayTeam, FTHG, FTAG, Date
        
        Returns:
            Training metrics (loss, accuracy, etc.)
        """
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """
        Generate predictions for matches.
        
        Args:
            data: DataFrame with upcoming matches & features
        
        Returns:
            List of Prediction objects
        """
        pass
    
    def predict_match(
        self, 
        home_team: str, 
        away_team: str,
        features: Optional[Dict[str, float]] = None,
        league: Optional[str] = None
    ) -> Prediction:
        """
        Predict single match. Default impl uses predict() on 1-row DF.
        Override for optimized single-match prediction.
        """
        df = pd.DataFrame([{
            "HomeTeam": home_team,
            "AwayTeam": away_team,
            "League": league or "unknown",
            **(features or {})
        }])
        predictions = self.predict(df)
        return predictions[0] if predictions else Prediction(
            match_id=f"{home_team}_vs_{away_team}",
            market=self.markets[0]
        )
    
    def supports_market(self, market: Market) -> bool:
        """Check if model supports a specific market."""
        return market in self.markets
    
    def save(self, path: Path):
        """Save model to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            "name": self.name,
            "markets": [m.value for m in self.markets],
            "is_fitted": self.is_fitted,
            "metadata": self.metadata,
            "model_state": self._get_state(),
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Load model from file (supporting both pickle and split-file formats)."""
        path = Path(path)
        
        # 1. Try Split-File Format (JSON Config)
        # Check if path is a directory or if path_config.json exists
        # Our save logic uses: parent / f"{stem}_config.json"
        config_path = path.parent / f"{path.stem}_config.json"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                
                name = config.get("name", "")
                
                # Dynamic factory
                if "BTTS" in name:
                    from stavki.models.gradient_boost.btts_model import BTTSModel
                    return BTTSModel.load(path) # Class-specific load (if implemented) or fallback
                elif "Neural" in name or "MultiTask" in name:
                    from stavki.models.neural.multitask import MultiTaskModel
                    return MultiTaskModel.load(path)
                
                # If other models adopt this format but don't have custom .load, we need generic logic here
                # But for now, only Neural uses it.
                
            except Exception as e:
                logger.warning(f"Found config at {config_path} but failed to load: {e}")
                # Fallthrough to try pickle
        
        # 2. Try Legacy Pickle
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path} (checked .pkl and _config.json)")
            
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        # If state is a dict (new save format), reconstruct object
        if isinstance(state, dict):
            name = state.get("name", "")
            
            # Dynamic factory based on name
            if "BTTS" in name:
                from stavki.models.gradient_boost.btts_model import BTTSModel
                model_cls = BTTSModel
            elif "CatBoost" in name:
                from stavki.models.catboost.catboost_model import CatBoostModel
                model_cls = CatBoostModel
            elif "LightGBM" in name:
                from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
                model_cls = LightGBMModel
            elif "Neural" in name or "MultiTask" in name:
                from stavki.models.neural.multitask import MultiTaskModel
                model_cls = MultiTaskModel
            elif "Poisson" in name or "Dixon" in name:
                from stavki.models.poisson.dixon_coles import DixonColesModel
                model_cls = DixonColesModel
            else:
                # Fallback: try to guess or use base if possible (though base is abstract)
                logger.warning(f"Unknown model type for {name}, trying to load as is.")
                # This might fail if cls is BaseModel
                model_cls = cls

            
            try:
                # Instantiate with default args
                # Models like CatBoostModel/LightGBMModel don't take name/markets in init
                instance = model_cls()
                
                # Restore base attributes
                instance.name = name
                instance.markets = [Market(m) for m in state.get("markets", [])]
                if not instance.markets:
                    instance.markets = [Market.MATCH_WINNER]
                
                instance.is_fitted = state.get("is_fitted", False)
                instance.metadata = state.get("metadata", {})
                
                # Restore internal state
                if "model_state" in state:
                    instance._set_state(state["model_state"])
                
                return instance
            except Exception as e:
                logger.error(f"Failed to reconstruct model {name}: {e}")
                raise e

        # Legacy: pickle might return the object directly
        return state
    
    @abstractmethod
    def _get_state(self) -> Dict[str, Any]:
        """Get internal state for serialization."""
        pass
    
    @abstractmethod
    def _set_state(self, state: Dict[str, Any]):
        """Restore internal state from serialization."""
        pass
    
    def __repr__(self):
        markets_str = ", ".join(m.value for m in self.markets)
        return f"{self.__class__.__name__}(name='{self.name}', markets=[{markets_str}], fitted={self.is_fitted})"


class CalibratedModel(BaseModel):
    """
    Base class for models with probability calibration.
    Adds Isotonic Regression or Temperature Scaling.
    """
    
    def __init__(self, name: str, markets: List[Market], calibration_method: str = "isotonic"):
        super().__init__(name, markets)
        self.calibration_method = calibration_method
        self.calibrators: Dict[str, Any] = {}  # Per outcome
        self.is_calibrated = False
    
    def fit_calibration(self, data: pd.DataFrame, predictions: List[Prediction]):
        """
        Fit calibration on holdout set.
        
        Args:
            data: DataFrame with actual outcomes (requires FTHG, FTAG columns)
            predictions: Model predictions on same data
        """
        from sklearn.isotonic import IsotonicRegression
        
        # Build a {match_id: row} lookup for actual outcomes
        actuals_lookup = {}
        for idx, row in data.iterrows():
            match_id = row.get(
                "match_id",
                f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}_{idx}"
            )
            actuals_lookup[match_id] = row
        
        # Collect predictions and actuals per outcome
        calibration_data: Dict[str, Dict[str, list]] = {}
        
        for pred in predictions:
            row = actuals_lookup.get(pred.match_id)
            if row is None:
                continue
            
            # Determine actual outcome for this market
            actual_outcome = self._resolve_actual_outcome(row, pred.market)
            if actual_outcome is None:
                continue
            
            for outcome, prob in pred.probabilities.items():
                key = f"{pred.market.value}_{outcome}"
                if key not in calibration_data:
                    calibration_data[key] = {"probs": [], "actuals": []}
                
                calibration_data[key]["probs"].append(prob)
                # 1.0 if this outcome actually occurred, 0.0 otherwise
                calibration_data[key]["actuals"].append(
                    1.0 if outcome == actual_outcome else 0.0
                )
        
        # Fit calibrators
        for key, data_dict in calibration_data.items():
            if len(data_dict["probs"]) > 10:  # Minimum samples
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(
                    np.array(data_dict["probs"]),
                    np.array(data_dict["actuals"]),
                )
                self.calibrators[key] = calibrator
        
        self.is_calibrated = True
        logger.info(
            f"Calibration fitted on {len(calibration_data)} outcome keys, "
            f"{sum(1 for v in self.calibrators.values() if hasattr(v, 'predict'))} calibrators active"
        )
    
    @staticmethod
    def _resolve_actual_outcome(row: pd.Series, market: Market) -> Optional[str]:
        """Determine actual outcome from a data row for a given market."""
        try:
            fthg = row.get("FTHG")
            ftag = row.get("FTAG")
            if fthg is None or ftag is None:
                return None
            fthg, ftag = int(fthg), int(ftag)
        except (ValueError, TypeError):
            return None
        
        if market == Market.MATCH_WINNER:
            if fthg > ftag:
                return "home"
            elif fthg < ftag:
                return "away"
            else:
                return "draw"
        elif market == Market.OVER_UNDER:
            return "over_2.5" if (fthg + ftag) > 2.5 else "under_2.5"
        elif market == Market.BTTS:
            return "yes" if (fthg > 0 and ftag > 0) else "no"
        return None
    
    def calibrate(self, predictions: List[Prediction]) -> List[Prediction]:
        """Apply calibration to predictions."""
        if not self.is_calibrated:
            return predictions
        
        calibrated = []
        for pred in predictions:
            new_probs = {}
            for outcome, prob in pred.probabilities.items():
                key = f"{pred.market.value}_{outcome}"
                if key in self.calibrators and hasattr(self.calibrators[key], "predict"):
                    # IsotonicRegression.predict expects a 1D array
                    new_probs[outcome] = float(self.calibrators[key].predict([prob])[0])
                else:
                    new_probs[outcome] = prob
            
            # Renormalize
            total = sum(new_probs.values())
            if total > 0:
                new_probs = {k: v/total for k, v in new_probs.items()}
            
            calibrated.append(Prediction(
                match_id=pred.match_id,
                market=pred.market,
                timestamp=pred.timestamp,
                probabilities=new_probs,
                confidence=pred.confidence,
                model_name=pred.model_name + "_calibrated",
                features_used=pred.features_used,
            ))
        
        return calibrated


class PerLeagueModel(BaseModel):
    """
    Base class for models that train separate instances per league.
    Each league can have different patterns (home advantage, scoring rates).
    """
    
    def __init__(self, name: str, markets: List[Market]):
        super().__init__(name, markets)
        self.league_models: Dict[str, BaseModel] = {}
        self.default_model: Optional[BaseModel] = None
    
    @abstractmethod
    def create_league_model(self, league: str) -> BaseModel:
        """Factory method to create a model instance for a league."""
        pass
    
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """Train separate models per league."""
        if "League" not in data.columns:
            raise ValueError("Data must have 'League' column for PerLeagueModel")
        
        metrics = {}
        
        for league in data["League"].unique():
            league_data = data[data["League"] == league]
            
            if len(league_data) < 50:  # Minimum matches for training
                continue
            
            model = self.create_league_model(league)
            league_metrics = model.fit(league_data, **kwargs)
            
            self.league_models[league] = model
            metrics[league] = league_metrics
        
        # Train default model on all data for unseen leagues
        self.default_model = self.create_league_model("default")
        self.default_model.fit(data, **kwargs)
        
        self.is_fitted = True
        return metrics
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Predict using league-specific models."""
        predictions = []
        
        for idx, row in data.iterrows():
            league = row.get("League", "unknown")
            model = self.league_models.get(league, self.default_model)
            
            if model:
                pred = model.predict(data.iloc[[idx]])
                predictions.extend(pred)
        
        return predictions
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "league_models": {
                league: model._get_state() 
                for league, model in self.league_models.items()
            },
            "default_model": self.default_model._get_state() if self.default_model else None,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        """Restore league models from saved state using the factory method."""
        league_states = state.get("league_models", {})
        for league, model_state in league_states.items():
            model = self.create_league_model(league)
            model._set_state(model_state)
            model.is_fitted = True
            self.league_models[league] = model
        
        default_state = state.get("default_model")
        if default_state is not None:
            self.default_model = self.create_league_model("default")
            self.default_model._set_state(default_state)
            self.default_model.is_fitted = True
