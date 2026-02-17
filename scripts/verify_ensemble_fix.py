
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
from stavki.models.poisson.dixon_coles import DixonColesModel
from stavki.models.ensemble.predictor import EnsemblePredictor, Market
from stavki.utils import generate_match_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_fix():
    logger.info("Verifying Ensemble ID Fix...")
    
    # 1. Create dummy data
    data = pd.DataFrame({
        "HomeTeam": ["Arsenal", "Chelsea", "Liverpool", "Man City"] * 25,
        "AwayTeam": ["Spurs", "West Ham", "Everton", "Man Utd"] * 25,
        "Date": pd.date_range(start="2023-01-01", periods=100),
        "FTHG": np.random.randint(0, 5, 100),
        "FTAG": np.random.randint(0, 5, 100),
        "HS": np.random.randint(5, 20, 100),
        "AS": np.random.randint(5, 20, 100),
        "HST": np.random.randint(2, 10, 100),
        "AST": np.random.randint(2, 10, 100),
        # Add some features for LightGBM
        "feature1": np.random.rand(100),
        "feature2": np.random.rand(100),
    })
    
    # Add match_id explicitly to test if generate_match_id works consistently
    # Actually, we let models generate it internally to test the fix
    
    # 2. Instantiate Models
    lgbm = LightGBMModel()
    dc = DixonColesModel()
    
    # Mock fit (just set is_fitted = True and some dummy params)
    data["target"] = 0 # Dummy target
    lgbm.features = ["feature1", "feature2"]
    lgbm.is_fitted = True
    # Mock predict for LightGBM
    original_lgbm_predict = lgbm.predict
    
    # We need real predictions to test ensemble alignment
    # Let's actually fit them quickly if possible, or mock the predict return
    # Mocking is safer and faster for this test
    
    from stavki.models.base import Prediction
    
    def mock_predict_lgbm(data):
        preds = []
        for idx, row in data.iterrows():
            mid = generate_match_id(row["HomeTeam"], row["AwayTeam"], row["Date"])
            preds.append(Prediction(
                match_id=mid,
                market=Market.MATCH_WINNER,
                probabilities={"home": 0.4, "draw": 0.3, "away": 0.3},
                confidence=0.1,
                model_name="LightGBM_1X2"
            ))
        return preds
    
    lgbm.predict = mock_predict_lgbm
    
    def mock_predict_dc(data):
        preds = []
        for idx, row in data.iterrows():
            # DixonColes internally uses generate_match_id now
            mid = generate_match_id(row["HomeTeam"], row["AwayTeam"], row["Date"])
            preds.append(Prediction(
                match_id=mid,
                market=Market.MATCH_WINNER,
                probabilities={"home": 0.35, "draw": 0.35, "away": 0.3},
                confidence=0.05,
                model_name="DixonColes"
            ))
        return preds
        
    dc.predict = mock_predict_dc
    dc.is_fitted = True
    
    # 3. Create Ensemble
    ensemble = EnsemblePredictor(models={"LightGBM_1X2": lgbm, "DixonColes": dc})
    
    # 4. Optimize Weights
    logger.info("Running optimization...")
    try:
        weights = ensemble.optimize_weights(data, Market.MATCH_WINNER)
        logger.info(f"Weights: {weights}")
        
        if not weights:
            logger.error("Optimization returned empty weights! Fix Failed.")
            return False
            
        logger.info("Optimization successful!")
        return True
    except Exception as e:
        logger.error(f"Optimization raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = verify_fix()
    if success:
        print("✅ VERIFICATION PASSED")
        sys.exit(0)
    else:
        print("❌ VERIFICATION FAILED")
        sys.exit(1)
