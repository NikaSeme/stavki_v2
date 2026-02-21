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

from scipy.optimize import minimize

from ..base import BaseModel, Prediction, Market, MatchPredictions

logger = logging.getLogger(__name__)


# Default ensemble weights per market
DEFAULT_WEIGHTS = {
    Market.MATCH_WINNER.value: {
        "CatBoost_1X2": 1.00,
        "DixonColes": 0.00,
        "LightGBM_1X2": 0.00,
        "NeuralMultiTask": 0.00,
        "DeepInteraction": 0.00,
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
        
        # Shadow models (Watcher Only) - Not used in weighted average
        self.shadow_models: Dict[str, BaseModel] = {}
        
        # Track model performance for adaptive weighting
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
    
    def add_model(self, model: BaseModel):
        """Add a model to the ensemble."""
        self.models[model.name] = model
        logger.info(f"Added model: {model.name}")

    def add_shadow_model(self, model: BaseModel):
        """Add a shadow (watcher) model."""
        self.shadow_models[model.name] = model
        logger.info(f"Added shadow model: {model.name}")
    
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
        """Generate ensemble predictions using vectorized operations."""
        all_predictions = []
        
        # 1. Collect predictions from all models
        # Structure: market -> match_id -> model_name -> Prediction
        market_preds: Dict[Market, Dict[str, Dict[str, Prediction]]] = defaultdict(lambda: defaultdict(dict))
        
        # ALIASING: Map pipeline columns to training columns
        # Training data uses PascalCase, pipeline uses snake_case
        df = data.copy()
        mappings = {
            "home_team": "HomeTeam",
            "away_team": "AwayTeam", 
            "league": "League",
            "date": "Date",
            "commence_time": "Date",
            # Map generic odds to specific columns models might expect (Avg/Max)
            "home_odds": "AvgH",
            "draw_odds": "AvgD",
            "away_odds": "AvgA",
        }
        for src, dst in mappings.items():
            if src in df.columns and dst not in df.columns:
                df[dst] = df[src]
            # Also map reverse if needed? No, pipeline has src.
            
        # Ensure Date is datetime
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        for name, model in self.models.items():
            if model.is_fitted:
                # Iterate over all markets this ensemble supports
                for market in self.markets:
                    if not model.supports_market(market):
                        continue
                    
                    try:
                        # Prepare data subset for this model (same code as before)
                        # ... (omitted for brevity, assume data preparation is fast or handled inside model)
                        # Ideally, models should handle extra columns gracefully.
                        # For now, we trust basic data compatibility or rely on model's internal handling.
                        
                        logger.debug(f"Predicting with {name} for {market.value}")
                        preds = model.predict(df)
                        logger.debug(f"  → {name} returned {len(preds)} predictions")
                        
                        for p in preds:
                            if p.market == market:
                                market_preds[market][p.match_id][name] = p
                                
                    except Exception as e:
                        logger.warning(f"Model {name} prediction for {market.value} failed: {e}")
            else:
                logger.debug(f"Model {name} is not fitted, skipping")

        # --- Shadow Models Execution ---
        for name, model in self.shadow_models.items():
            if model.is_fitted:
                 for market in self.markets:
                    if not model.supports_market(market): continue
                    try:
                        logger.info(f"Running shadow prediction: {name} ({market.value})")
                        # Shadow models handle their own logging/persistence
                        _ = model.predict(df) 
                    except Exception as e:
                        logger.warning(f"Shadow model {name} failed: {e}")

        
        # 2. Build League Lookup Vectorized
        league_lookup = {}
        # df has aliased columns (PascalCase)
        league_col = "League"
        
        home_col = "HomeTeam" if "HomeTeam" in df.columns else "home_team"
        away_col = "AwayTeam" if "AwayTeam" in df.columns else "away_team"
        date_col = "Date" if "Date" in df.columns else "commence_time"
        
        if league_col in df.columns and home_col in df.columns:
            # Try to vectorize ID generation if possible, else use apply
            from stavki.utils import generate_match_id
            
            # Use a temporary dataframe to avoid modifying input
            temp = df[[league_col, home_col, away_col, date_col]].copy()
            temp["mid"] = temp.apply(
                lambda x: generate_match_id(str(x.get(home_col, "")), str(x.get(away_col, "")), str(x.get(date_col, ""))), 
                axis=1
            )
            league_lookup = dict(zip(temp["mid"], temp[league_col]))
            
        # 3. Process each market vectorized
        for market, match_dict in market_preds.items():
            # Convert to list for processing
            match_ids = list(match_dict.keys())
            
            if not match_ids:
                continue
                
            # Get outcomes from first prediction
            first_match = match_ids[0]
            first_model = list(match_dict[first_match].keys())[0]
            outcomes = sorted(list(match_dict[first_match][first_model].probabilities.keys()))
            outcome_map = {out: i for i, out in enumerate(outcomes)}
            n_outcomes = len(outcomes)
            
            # Build Tensors
            # We need to handle missing models for some matches
            # But efficiently.
            
            # Let's iterate matches to create Prediction objects directly if N is small?
            # No, goal is vectorization.
            
            # Since models might differ per match, strict vectorization is hard 
            # unless we align everything.
            # But the weighting logic is what's slow.
            
            for match_id in match_ids:
                preds_map = match_dict[match_id]
                league = league_lookup.get(match_id)
                weights = self.get_weights(market, league)
                
                # Fast inner loop
                valid_preds = []
                valid_weights = []
                
                for model_name, pred in preds_map.items():
                    w = weights.get(model_name, 0.0)
                    if w > 0:
                        valid_preds.append(pred)
                        valid_weights.append(w)
                    # else:
                        # print(f"DEBUG: Skipping {model_name} due to 0 weight")
                
                if not valid_preds:
                    continue
                    
                # Normalize weights
                total_w = sum(valid_weights)
                if total_w == 0:
                    probs = {o: 0.0 for o in outcomes} # Should not happen
                    confidence = 0.0
                else:
                    norm_weights = [w/total_w for w in valid_weights]
                    
                    # Weighted Sum
                    final_probs = {o: 0.0 for o in outcomes}
                    for i, p in enumerate(valid_preds):
                        nw = norm_weights[i]
                        for o, prob in p.probabilities.items():
                            final_probs[o] += prob * nw
                    
                    # Confidence
                    sorted_p = sorted(final_probs.values(), reverse=True)
                    confidence = sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0]
                    
                    # Disagreement (only if enabled)
                    disagreement = 0.0
                    if self.use_disagreement and len(valid_preds) >= 2:
                        # Vectorized JS divergence for this single match
                        # Build (N_models, N_outcomes) matrix
                        p_matrix = np.zeros((len(valid_preds), n_outcomes))
                        for i, p in enumerate(valid_preds):
                            for o, prob in p.probabilities.items():
                                if o in outcome_map:
                                    p_matrix[i, outcome_map[o]] = prob
                        
                        # Mean distribution
                        m = np.mean(p_matrix, axis=0)
                        
                        # KL Divergence: sum(p * log(p/m))
                        # Add epsilon
                        eps = 1e-10
                        # p * np.log((p+eps)/(m+eps))
                        kls = np.sum(p_matrix * np.log((p_matrix + eps) / (m + eps)), axis=1)
                        disagreement = np.mean(kls)

                    probs = final_probs
                    
                    # Create Prediction
                    all_predictions.append(Prediction(
                        match_id=match_id,
                        market=market,
                        probabilities=probs,
                        confidence=confidence * (1 - disagreement * 0.5),
                        model_name=self.name,
                        features_used={"disagreement": disagreement, "n_models": len(valid_preds)}
                    ))
                    
        return all_predictions
    

    
    def _build_league_lookup(self, data: pd.DataFrame) -> Dict[str, str]:
        """Build a {match_id: league} lookup dict from the data."""
        lookup: Dict[str, str] = {}
        
        # Check for column (case insensitive priority)
        league_col = "League" if "League" in data.columns else "league"
        if league_col not in data.columns:
            return lookup
        
        for idx, row in data.iterrows():
            from stavki.utils import generate_match_id
            match_id = row.get(
                "match_id",
                generate_match_id(row.get('HomeTeam', row.get('home_team', 'home')), row.get('AwayTeam', row.get('away_team', 'away')), row.get('Date', row.get('commence_time')))
            )
            league = row.get(league_col)
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
        Optimize ensemble weights using SLSQP minimization.
        
        Args:
            data: Validation DataFrame with actual outcomes
            market: Market to optimize for
            league: Optional league-specific optimization
            metric: Metric to optimize ('brier' or 'log_loss')
        
        Returns:
            Optimized weights
        """
        # 1. Collect predictions from all models
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
        
        # 2. Vectorize data for fast optimization
        # matrix shape: (n_samples, n_models, n_outcomes)
        # targets shape: (n_samples, n_outcomes) [one-hot]
        try:
            prediction_matrix, target_matrix, outcomes = self._vectorize_for_optimization(
                data, model_predictions, market
            )
        except ValueError as e:
            logger.warning(f"Optimization failed: {e}")
            return {}
            
        if len(prediction_matrix) < 10:
            logger.warning("Not enough samples for optimization")
            return {}

        # 3. Define objective function
        def objective(weights):
            # weights: (n_models,)
            # pred_matrix: (n_samples, n_models, n_outcomes)
            # weighted: (n_samples, n_outcomes)
            weighted_probs = np.tensordot(prediction_matrix, weights, axes=([1], [0]))
            
            # Normalize (just in case, though constraint handles sum=1)
            # Avoid div by zero
            row_sums = weighted_probs.sum(axis=1, keepdims=True)
            weighted_probs = np.divide(weighted_probs, row_sums, where=row_sums!=0)
            
            if metric == "log_loss":
                # Clip for numerical stability
                probs = np.clip(weighted_probs, 1e-15, 1 - 1e-15)
                # Cross-entropy: -sum(y_true * log(y_pred))
                return -np.mean(np.sum(target_matrix * np.log(probs), axis=1))
            else:
                # Brier score: mean((y_pred - y_true)^2)
                return np.mean(np.sum((weighted_probs - target_matrix)**2, axis=1))

        # 4. Run optimization
        # Initial guess: equal weights
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: sum(weights) = 1
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        # Bounds: 0 <= w <= 1
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        
        result = minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            tol=1e-6,
        )
        
        if not result.success:
            logger.warning(f"Optimization failed: {result.message}")
            return {name: 1.0/n_models for name in model_names}
            
        # 5. Extract results
        optimized_weights = dict(zip(model_names, result.x))
        
        # Clean up small weights
        optimized_weights = {k: v if v > 0.001 else 0.0 for k, v in optimized_weights.items()}
        
        # Re-normalize
        total = sum(optimized_weights.values())
        if total > 0:
            optimized_weights = {k: v/total for k, v in optimized_weights.items()}
            
        # Set weights
        self.set_weights(market, optimized_weights, league)
        
        logger.info(f"Optimized weights for {market.value}: {optimized_weights}")
        logger.info(f"Final {metric}: {result.fun:.4f}")
        
        return optimized_weights

    def _vectorize_for_optimization(
        self,
        data: pd.DataFrame,
        model_predictions: Dict[str, List[Prediction]],
        market: Market
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Convert predictions to numpy arrays for vectorization.
        
        Returns:
            prediction_matrix: (n_samples, n_models, n_outcomes)
            target_matrix: (n_samples, n_outcomes) one-hot encoded
            outcomes: List of outcome names
        """
        # Align samples
        common_ids = set()
        first_model = list(model_predictions.keys())[0]
        common_ids = {p.match_id for p in model_predictions[first_model]}
        
        for name in model_predictions:
            ids = {p.match_id for p in model_predictions[name]}
            common_ids.intersection_update(ids)
            
        sorted_ids = sorted(list(common_ids))
        
        if not sorted_ids:
            raise ValueError("No common matches found across models")
            
        # Determine outcome space
        first_pred = model_predictions[first_model][0]
        outcomes = sorted(list(first_pred.probabilities.keys()))
        outcome_map = {out: i for i, out in enumerate(outcomes)}
        n_outcomes = len(outcomes)
        
        # Build matrices
        n_samples = len(sorted_ids)
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        
        pred_matrix = np.zeros((n_samples, n_models, n_outcomes))
        target_matrix = np.zeros((n_samples, n_outcomes))
        
        # Lookup actuals
        # Build lookup for data rows
        data_rows = {}
        for _, row in data.iterrows():
            from stavki.utils import generate_match_id
            mid = row.get("match_id", generate_match_id(row.get('HomeTeam', row.get('home_team', '')), row.get('AwayTeam', row.get('away_team', '')), row.get('Date', row.get('commence_time'))))
            # Handling generic ID matching if needed, but assuming exact match for speed
            # If standard ID generation is consistent, this works. 
            # If not, we might need a better join strategy.
            # Fallback to simple lookup if generic ID fails
            data_rows[mid] = row
            
            # Also try flexible matching if exact fails? 
            # For optimization speed, exact match is preferred.
        
        valid_indices = []
        
        for i, mid in enumerate(sorted_ids):
            # Get Targets
            # We need to find the data row for this match_id
            # This is tricky if match_ids are not in data or formats differ.
            # Assuming match_id is consistent.
            
            # Simple linear scan is too slow. Use the dict.
            row = data_rows.get(mid)
            if row is None:
                # Try finding by fuzzy match? 
                # Skip for now to assume consistency
                continue
                
            actual = self._get_actual_outcome(row, market)
            if actual is None or actual not in outcome_map:
                continue
                
            target_idx = outcome_map[actual]
            target_matrix[i, target_idx] = 1.0
            
            # Get Predictions
            for j, name in enumerate(model_names):
                # Find pred for this mid
                # Using a dict lookup for predictions of each model would be faster than list scan
                # But here we have lists.
                # Let's optimize: pre-convert lists to dicts
                pass 
            
            valid_indices.append(i)

        # Optimization: Pre-map all predictions to dicts
        pred_lookups = [
            {p.match_id: p for p in model_predictions[name]}
            for name in model_names
        ]
        
        # Re-run construction with Lookups
        final_valid_indices = []
        
        for i, mid in enumerate(sorted_ids):
            row = data_rows.get(mid)
            if row is None: continue
            
            actual = self._get_actual_outcome(row, market)
            if actual is None or actual not in outcome_map: continue
            
            # Fill Target
            target_idx = outcome_map[actual]
            target_matrix[i, target_idx] = 1.0
            
            # Fill Preds
            complete_data = True
            for j in range(n_models):
                pred = pred_lookups[j].get(mid)
                if not pred:
                    complete_data = False
                    break
                
                for out_name, prob in pred.probabilities.items():
                    if out_name in outcome_map:
                        pred_matrix[i, j, outcome_map[out_name]] = prob
            
            if complete_data:
                final_valid_indices.append(i)
        
        if not final_valid_indices:
            raise ValueError("No valid samples after alignment")
            
        return (
            pred_matrix[final_valid_indices],
            target_matrix[final_valid_indices],
            outcomes
        )

    
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
        """Load weights from JSON file (league_config.json)."""
        if not path.exists():
            # Try default location if path doesn't exist but is just a filename
            if path.name == "league_config.json":
                # Try new standard name
                alt_path = path.parent / "league_weights.json"
                if alt_path.exists():
                    path = alt_path

        try:
            with open(path) as f:
                config = json.load(f)
            
            # Support both new structure (league -> market -> weights) and legacy
            # New structure: {"EPL": {"1x2": {"model": 0.5}}}
            # Legacy: {"global": ..., "per_league": ...}
            
            if "global" in config:
                self.weights = config["global"]
                self.league_weights = config.get("per_league", {})
            elif any(k in config for k in ["EPL", "La Liga", "Serie A", "epl", "laliga"]):
                # Assumed to be pure per-league weights file
                # Check structure
                first_key = list(config.keys())[0]
                if "1x2" in config[first_key] or "weights" in config[first_key]:
                     # It's a league weights file
                     self.league_weights = config
                else:
                    # Might be legacy leagues.json with just integers?
                    pass
            else:
                self.weights = config
                self.league_weights = {}
                
            logger.info(f"Loaded ensemble weights from {path.name}")
            
        except Exception as e:
            logger.error(f"Failed to load weights from {path}: {e}")
    
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
