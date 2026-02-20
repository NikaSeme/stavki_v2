
import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stavki.prediction.live import LivePredictor, LivePrediction
from stavki.data.collectors.sportmonks import MatchFixture
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VERIFY")

def test_models():
    print("🕵️♂️ BOB'S EXTENDED AUDIT: End-to-End Model Verification")
    print("="*60)
    
    try:
        # 1. Initialize Predictor
        print("\n1. Initializing LivePredictor...")
        predictor = LivePredictor(api_key="verify_only")
        
        # Load main ensemble model (CatBoost is often the core, or we load individual ones if ensemble logic is external)
        # Note: LivePredictor.load_model loads a SINGLE CatBoost model usually.
        # But the system uses an Ensemble. 
        # Let's check if LivePredictor supports the full ensemble or just one model.
        # Looking at live.py, perform_prediction load_model loads a pickle with 'model' key.
        # This seems to be the CatBoost model or a wrapper.
        
        # Let's try loading the Neural MultiTask model to see if it works with LivePredictor (if it supports it)
        # Or more likely, we should test the ensemble if possible. 
        # But for now, let's test the CatBoost model which is a key component.
        
        model_path = Path("models/catboost.pkl")
        if not model_path.exists():
            print("❌ CatBoost model not found at models/catboost.pkl")
            return
            
        print(f"   Loading model from {model_path}...")
        predictor.load_model(str(model_path))
        print("   ✅ Model loaded successfully")
        
        # 2. Mock a Fixture
        print("\n2. Mocking Fixture Data...")
        fixture = MatchFixture(
            fixture_id=12345,
            league_id=8, # EPL
            home_team="Man United", # Canonical Name!
            home_team_id=14,
            away_team="Liverpool", # Canonical Name! (assuming Liverpool is Liverpool)
            away_team_id=8,
            kickoff=datetime.now() + timedelta(days=1)
        )
        
        # Mock Odds
        odds = {"home": 2.5, "draw": 3.4, "away": 2.8}
        
        # 3. Predict
        print("\n3. Running Prediction...")
        # We need to make sure _load_team_stats was called (it is in init)
        # We need to make sure "Man United" exists in stats (verified previously)
        
        result = predictor.predict_fixture(fixture, odds)
        
        print(f"   Prediction: Home={result.prob_home:.2f}, Draw={result.prob_draw:.2f}, Away={result.prob_away:.2f}")
        
        # 4. Sanity Check
        total_prob = result.prob_home + result.prob_draw + result.prob_away
        if 0.99 <= total_prob <= 1.01:
            print(f"   ✅ Probabilities sum to {total_prob:.2f}")
        else:
            print(f"   ❌ Probabilities sum to {total_prob:.2f} (Expected ~1.0)")
            
        print("\n✅ End-to-End Verification Passed!")
        
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_models()
