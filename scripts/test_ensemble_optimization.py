
import pandas as pd
import numpy as np
import logging
from unittest.mock import MagicMock
from stavki.models.ensemble.predictor import EnsemblePredictor, Market
from stavki.models.base import BaseModel, Prediction


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockModel(BaseModel):
    def __init__(self, name, predictions_map):
        super().__init__(name=name, markets=[Market.MATCH_WINNER])
        self.predictions_map = predictions_map
        self.is_fitted = True

    def predict(self, data):
        return [self.predictions_map.get(row["match_id"]) for _, row in data.iterrows() if row["match_id"] in self.predictions_map]
    
    def supports_market(self, market):
        return True

    def fit(self, data, **kwargs):
        pass

    def _get_state(self):
        return {}

    def _set_state(self, state):
        pass


def test_ensemble_optimization():
    logger.info("Testing Ensemble Optimization...")
    
    # Create synthetic data
    matches = [f"Match_{i}" for i in range(20)]
    outcomes = ["home", "draw", "draw", "away", "home"] * 4
    
    data = pd.DataFrame({
        "match_id": matches,
        "FTHG": [1, 1, 0, 0, 2] * 4,
        "FTAG": [0, 1, 0, 1, 1] * 4,
        "HomeTeam": ["Home"] * 20, # needed for get_actual_outcome if failing match_id lookup? 
        # Actually optimize_weights uses data rows.
        # But _vectorize uses match_id lookup first.
    })
    
    # Mock Predictions
    # Model A: Good at Home wins
    probs_A_base = {
        0: {"home": 0.8, "draw": 0.1, "away": 0.1}, # Correct (Home)
        1: {"home": 0.6, "draw": 0.2, "away": 0.2}, # Wrong (Draw)
        2: {"home": 0.3, "draw": 0.4, "away": 0.3}, # Correct (Draw)
        3: {"home": 0.2, "draw": 0.2, "away": 0.6}, # Correct (Away)
        4: {"home": 0.9, "draw": 0.05, "away": 0.05} # Correct (Home)
    }
    
    probs_A = {}
    for i in range(20):
        probs_A[matches[i]] = probs_A_base[i % 5]
    
    # Model B: Randomish
    probs_B = {
        m: {"home": 0.33, "draw": 0.33, "away": 0.34} for m in matches
    }

    
    # Create Prediction objects
    preds_A = {m: Prediction(match_id=m, market=Market.MATCH_WINNER, probabilities=probs_A[m], confidence=1.0, model_name="ModelA") for m in matches}
    preds_B = {m: Prediction(match_id=m, market=Market.MATCH_WINNER, probabilities=probs_B[m], confidence=1.0, model_name="ModelB") for m in matches}

    
    model_A = MockModel("ModelA", preds_A)
    model_B = MockModel("ModelB", preds_B)
    
    ensemble = EnsemblePredictor(models={"ModelA": model_A, "ModelB": model_B})
    
    # Run optimization
    weights = ensemble.optimize_weights(data, Market.MATCH_WINNER, metric="brier")
    
    logger.info(f"Optimized Weights: {weights}")
    
    # Assertions
    assert "ModelA" in weights
    assert "ModelB" in weights
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    
    # Model A is much better, should have higher weight
    assert weights["ModelA"] > weights["ModelB"]
    
    logger.info("Verification Successful!")

if __name__ == "__main__":
    test_ensemble_optimization()
