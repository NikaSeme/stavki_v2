"""
Ensemble Predictor - Combines Multiple Models
==============================================

Weighted ensemble with:
- Per-league optimized weights
- Dynamic disagreement detection
- Confidence-based blending
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import json
import logging
from pathlib import Path

from ..base import BaseModel, Prediction, Market, MatchPredictions

logger = logging.getLogger(__name__)


# Default ensemble weights per market
DEFAULT_WEIGHTS = {
    Market.MATCH_WINNER.value: {
        "DixonColes": 0.25,
        "LightGBM_1X2": 0.35,
        "NeuralMultiTask": 0.30,
        "MarketAdjuster": 0.10,
    },
    Market.OVER_UNDER.value: {
        "DixonColes": 0.30,
        "GoalsRegressor": 0.25,
        "NeuralMultiTask": 0.25,
        "LightGBM_1X2": 0.10,
        "MarketAdjuster": 0.10,
    },
    Market.BTTS.value: {
        "LightGBM_BTTS": 0.35,
        "DixonColes": 0.25,
        "NeuralMultiTask": 0.25,
        "GoalsRegressor": 0.10,
        "MarketAdjuster": 0.05,
    },
}


class EnsemblePredictor(BaseModel):
    """
    Ensemble predictor combining multiple base models.
    
    Features:
    - Per-market weights
    - Per-league weight optimization
    - Disagreement score calculation
    - Confidence-weighted averaging
    """
    
    def __init__(
        self,
        models: Optional[Dict[str, BaseModel]] = None,
        weights: Optional[Dict[str, Dict[str, float]]] = None,
        league_weights: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        use_disagreement: bool = True,
    ):
        super().__init__(
            name="Ensemble",
            markets=[Market.MATCH_WINNER, Market.OVER_UNDER, Market.BTTS]
        )
        
        self.models: Dict[str, BaseModel] = models or {}
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.league_weights = league_weights or {}  # league -> market -> model -> weight
        self.use_disagreement = use_disagreement
        
        # Track model performance for adaptive weighting
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_model(self, model: BaseModel):
        """Add a model to the ensemble."""
        self.models[model.name] = model
        logger.info(f"Added model: {model.name}")
    
    def remove_model(self, name: str):
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]
            logger.info(f"Removed model: {name}")
    
    def set_weights(
        self, 
        market: Market, 
        weights: Dict[str, float],
        league: Optional[str] = None
    ):
        """Set weights for a specific market (optionally per-league)."""
        # Normalize weights
        total = sum(weights.values())
        normalized = {k: v/total for k, v in weights.items()}
        
        if league:
            if league not in self.league_weights:
                self.league_weights[league] = {}
            self.league_weights[league][market.value] = normalized
        else:
            self.weights[market.value] = normalized
    
    def get_weights(self, market: Market, league: Optional[str] = None) -> Dict[str, float]:
        """Get weights for a market, using league-specific if available."""
        if league and league in self.league_weights:
            league_specific = self.league_weights[league].get(market.value)
            if league_specific:
                return league_specific
        
        return self.weights.get(market.value, {})
    
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train all component models and optimize ensemble weights.
        
        Note: This trains individual models. For weight optimization,
        use optimize_weights() on a validation set.
        """
        metrics = {}
        
        for name, model in self.models.items():
            if not model.is_fitted:
                logger.info(f"Training {name}...")
                try:
                    model_metrics = model.fit(data, **kwargs)
                    metrics[name] = model_metrics
                except Exception as e:
                    logger.error(f"Failed to train {name}: {e}")
        
        self.is_fitted = True
        return metrics
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate ensemble predictions."""
        all_predictions = []
        
        # Collect predictions from all models
        model_predictions: Dict[str, List[Prediction]] = {}
        
        for name, model in self.models.items():
            if model.is_fitted:
                # Iterate over all markets this ensemble supports
                for market in self.markets:
                    if not model.supports_market(market):
                        continue
                    
                    try:
                        # Prepare data subset for this model
                        model_features = getattr(model, "features", [])
                        if not model_features and hasattr(model, "metadata"):
                            model_features = model.metadata.get("features", [])
                        
                        # If model has specific features, enforce them
                        if model_features and len(model_features) > 0:
                            # Check if we have all features
                            missing = [f for f in model_features if f not in data.columns]
                            if missing:
                                # Log warning but try to proceed (defaults might be handled in predict)
                                logger.debug(f"Model {name} missing features: {len(missing)}")
                            
                            # Create subset with ONLY model features + meta
                            cols_to_use = [f for f in model_features if f in data.columns]
                            
                            # Add essential meta columns for prediction matching
                            meta_cols = ["match_id", "HomeTeam", "AwayTeam", "Date", "League"]
                            # Handle categorical features if present
                            model_cat_features = getattr(model, "cat_features", [])
                            if model_cat_features:
                                # Helper for fuzzy matching (ignore case and underscores)
                                def normalize(s):
                                    return s.lower().replace("_", "")

                                for cf in model_cat_features:
                                    if cf in data.columns and cf not in cols_to_use:
                                        cols_to_use.append(cf)
                                    elif cf == "league" and "League" in data.columns:
                                        # Special case for league/League mismatch
                                        if "League" not in cols_to_use:
                                            cols_to_use.append("League")
                                    else:
                                        # Fuzzy match: HomeTeam matches home_team
                                        found = False
                                        for dc in data.columns:
                                            if normalize(dc) == normalize(cf) and dc not in cols_to_use:
                                                cols_to_use.append(dc)
                                                found = True
                                                break
                                        if found: 
                                            continue

                                model_data = data[cols_to_use].copy()
                                
                                # Rename columns to match model expectations
                                # This handles League->league, home_team->HomeTeam, etc.
                                rename_map = {}
                                if "League" in model_data.columns and "league" in model_cat_features:
                                    rename_map["League"] = "league"
                                
                                # General fuzzy rename
                                for cf in model_cat_features:
                                    if cf not in model_data.columns:
                                        for col in model_data.columns:
                                            if normalize(col) == normalize(cf):
                                                rename_map[col] = cf
                                                break
                                
                                if rename_map:
                                    model_data = model_data.rename(columns=rename_map)
                                    
                                # DEBUG LOGGING for CatBoost failure
                                if "CatBoost" in name:
                                    logger.info(f"DEBUG: CatBoost model_data shape: {model_data.shape}")
                                    logger.info(f"DEBUG: CatBoost model_data columns: {model_data.columns.tolist()}")
                                    logger.info(f"DEBUG: Missing cats? names: {model_cat_features}")
                                    missing_cats = [c for c in model_cat_features if c not in model_data.columns]
                                    if missing_cats:
                                        logger.error(f"DEBUG: CRITICAL - Missing cat features: {missing_cats}")
                                        logger.info(f"DEBUG: Source data columns: {data.columns.tolist()}")
                            else:
                                model_data = data[cols_to_use].copy()
                        
                        preds = model.predict(model_data)
                        # Store predictions for each market separately
                        if name not in model_predictions:
                            model_predictions[name] = []
                        model_predictions[name].extend(preds)
                    except Exception as e:
                        logger.warning(f"Model {name} prediction for market {market.value} failed: {e}")
        
        # Group by match and market
        grouped = self._group_predictions(model_predictions, data)
        
        # Pre-build league lookup (avoids O(n²) row scanning)
        league_lookup = self._build_league_lookup(data)
        
        # Ensemble each group
        for (match_id, market), preds_dict in grouped.items():
            league = league_lookup.get(match_id)
            
            ensemble_pred = self._ensemble_predictions(
                preds_dict, 
                market, 
                match_id,
                league
            )
            
            if ensemble_pred:
                all_predictions.append(ensemble_pred)
        
        return all_predictions
    
    def _group_predictions(
        self, 
        model_predictions: Dict[str, List[Prediction]],
        data: pd.DataFrame
    ) -> Dict[Tuple[str, Market], Dict[str, Prediction]]:
        """Group predictions by (match_id, market)."""
        grouped = defaultdict(dict)
        
        for model_name, predictions in model_predictions.items():
            for pred in predictions:
                key = (pred.match_id, pred.market)
                grouped[key][model_name] = pred
        
        return grouped
    
    def _ensemble_predictions(
        self,
        predictions: Dict[str, Prediction],
        market: Market,
        match_id: str,
        league: Optional[str] = None,
    ) -> Optional[Prediction]:
        """Combine predictions from multiple models."""
        if not predictions:
            return None
        
        # Get weights for this market
        weights = self.get_weights(market, league)
        
        # Collect all possible outcomes
        all_outcomes = set()
        for pred in predictions.values():
            all_outcomes.update(pred.probabilities.keys())
        
        # Initialize combined probabilities
        combined_probs = {outcome: 0.0 for outcome in all_outcomes}
        total_weight = 0.0
        
        # Weighted average
        for model_name, pred in predictions.items():
            weight = weights.get(model_name, 0.0)
            
            if weight > 0:
                for outcome, prob in pred.probabilities.items():
                    combined_probs[outcome] += prob * weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            combined_probs = {k: v/total_weight for k, v in combined_probs.items()}
        else:
            # Equal weights fallback
            n = len(predictions)
            for pred in predictions.values():
                for outcome, prob in pred.probabilities.items():
                    combined_probs[outcome] += prob / n
        
        # Calculate disagreement
        disagreement = self._calc_disagreement(predictions) if self.use_disagreement else 0.0
        
        # Confidence
        sorted_probs = sorted(combined_probs.values(), reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
        
        return Prediction(
            match_id=match_id,
            market=market,
            probabilities=combined_probs,
            confidence=confidence * (1 - disagreement * 0.5),  # Reduce confidence if disagreement
            model_name=self.name,
            features_used={"disagreement": disagreement, "n_models": len(predictions)},
        )
    
    def _calc_disagreement(self, predictions: Dict[str, Prediction]) -> float:
        """
        Calculate disagreement between models.
        
        Returns: Score 0-1, where 1 = complete disagreement
        """
        if len(predictions) < 2:
            return 0.0
        
        # Collect all probabilities as vectors
        prob_vectors = []
        
        for pred in predictions.values():
            # Sort by key for consistent ordering
            sorted_probs = [v for k, v in sorted(pred.probabilities.items())]
            prob_vectors.append(sorted_probs)
        
        # Pad to same length
        max_len = max(len(v) for v in prob_vectors)
        prob_vectors = [v + [0.0] * (max_len - len(v)) for v in prob_vectors]
        
        # Calculate pairwise Jensen-Shannon divergence
        prob_matrix = np.array(prob_vectors)
        
        disagreement = 0.0
        n_pairs = 0
        
        for i in range(len(prob_matrix)):
            for j in range(i + 1, len(prob_matrix)):
                p = prob_matrix[i]
                q = prob_matrix[j]
                m = (p + q) / 2
                
                # KL divergence with smoothing
                eps = 1e-10
                kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
                kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
                
                js = (kl_pm + kl_qm) / 2
                disagreement += js
                n_pairs += 1
        
        return float(disagreement / max(n_pairs, 1))
    
    def _build_league_lookup(self, data: pd.DataFrame) -> Dict[str, str]:
        """Build a {match_id: league} lookup dict from the data."""
        lookup: Dict[str, str] = {}
        if "League" not in data.columns:
            return lookup
        
        for idx, row in data.iterrows():
            match_id = row.get(
                "match_id",
                f"{row.get('HomeTeam', 'home')}_vs_{row.get('AwayTeam', 'away')}"
            )
            league = row.get("League")
            if match_id and league:
                lookup[match_id] = league
        
        return lookup
    
    def optimize_weights(
        self,
        data: pd.DataFrame,
        market: Market,
        league: Optional[str] = None,
        metric: str = "brier",
    ) -> Dict[str, float]:
        """
        Optimize ensemble weights on validation data using grid search.
        
        Args:
            data: Validation DataFrame with actual outcomes
            market: Market to optimize for
            league: Optional league-specific optimization
            metric: Metric to optimize ('brier', 'log_loss', 'accuracy')
        
        Returns:
            Optimized weights
        """
        # Get model predictions
        model_predictions = {}
        for name, model in self.models.items():
            if model.is_fitted and model.supports_market(market):
                try:
                    preds = model.predict(data)
                    model_predictions[name] = preds
                except Exception:
                    pass
        
        if len(model_predictions) < 2:
            logger.warning("Not enough models for optimization")
            return {}
        
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        
        # Dirichlet-sampled random search:
        # - All weight vectors automatically sum to 1.0
        # - Scales linearly with n_trials regardless of model count
        # - Concentration alpha=1.0 gives uniform distribution over simplex
        n_trials = 500
        patience = 100  # Stop early if no improvement for this many trials
        
        best_score = float("inf")
        best_weights = {}
        no_improvement = 0
        
        rng = np.random.default_rng(42)
        
        for trial in range(n_trials):
            # Sample from Dirichlet distribution (uniform over simplex)
            raw_weights = rng.dirichlet(np.ones(n_models))
            weight_dict = dict(zip(model_names, raw_weights))
            
            # Evaluate
            score = self._evaluate_weights(
                data, model_predictions, weight_dict, market, metric
            )
            
            if score < best_score:
                best_score = score
                best_weights = weight_dict
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= patience:
                logger.info(f"Early stop at trial {trial + 1}/{n_trials}")
                break
        
        # Set optimized weights
        if best_weights:
            self.set_weights(market, best_weights, league)
        
        logger.info(f"Optimized weights for {market.value}: {best_weights}")
        return best_weights
    
    def _evaluate_weights(
        self,
        data: pd.DataFrame,
        model_predictions: Dict[str, List[Prediction]],
        weights: Dict[str, float],
        market: Market,
        metric: str,
    ) -> float:
        """Evaluate a weight configuration."""
        grouped = self._group_predictions(model_predictions, data)
        
        scores = []
        
        for idx, row in data.iterrows():
            match_id = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}_{idx}")
            key = (match_id, market)
            
            if key not in grouped:
                continue
            
            preds_dict = grouped[key]
            
            # Compute weighted prediction
            combined = defaultdict(float)
            total_weight = 0.0
            
            for model_name, pred in preds_dict.items():
                w = weights.get(model_name, 0.0)
                for outcome, prob in pred.probabilities.items():
                    combined[outcome] += prob * w
                total_weight += w
            
            if total_weight == 0:
                continue
            
            combined = {k: v/total_weight for k, v in combined.items()}
            
            # Get actual outcome
            actual = self._get_actual_outcome(row, market)
            if actual is None:
                continue
            
            # Score
            if metric == "brier":
                score = sum((v - (1 if k == actual else 0))**2 for k, v in combined.items())
                scores.append(score)  # Lower = better calibration
            elif metric == "log_loss":
                prob = combined.get(actual, 0.01)
                scores.append(-np.log(max(prob, 1e-10)))
        
        return np.mean(scores) if scores else float("inf")
    
    def _get_actual_outcome(self, row: pd.Series, market: Market) -> Optional[str]:
        """Extract actual outcome from row."""
        if market == Market.MATCH_WINNER:
            if row["FTHG"] > row["FTAG"]:
                return "home"
            elif row["FTHG"] < row["FTAG"]:
                return "away"
            else:
                return "draw"
        elif market == Market.OVER_UNDER:
            return "over_2.5" if (row["FTHG"] + row["FTAG"]) > 2.5 else "under_2.5"
        elif market == Market.BTTS:
            return "yes" if (row["FTHG"] > 0 and row["FTAG"] > 0) else "no"
        return None
    
    def save_weights(self, path: Path):
        """Save current weights to JSON file."""
        weights_data = {
            "global": self.weights,
            "per_league": self.league_weights,
        }
        with open(path, "w") as f:
            json.dump(weights_data, f, indent=2)
    
    def load_weights(self, path: Path):
        """Load weights from JSON file."""
        with open(path) as f:
            weights_data = json.load(f)
        self.weights = weights_data.get("global", {})
        self.league_weights = weights_data.get("per_league", {})
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "league_weights": self.league_weights,
            "use_disagreement": self.use_disagreement,
            "model_performance": dict(self.model_performance),
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.weights = state["weights"]
        self.league_weights = state["league_weights"]
        self.use_disagreement = state["use_disagreement"]
        self.model_performance = defaultdict(dict, state["model_performance"])
