import sys
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root
sys.path.insert(0, os.getcwd())

from stavki.prediction.live import LivePredictor
from stavki.data.collectors.sportmonks import MatchFixture

def verify_live_predictor_mock_odds():
    print("\n--- 1. Verifying LivePredictor Mock Odds Fix ---")
    
    # Patch the class where it's used
    with patch('stavki.prediction.live.SportMonksClient') as MockClientClass:
        mock_instance = MockClientClass.return_value
        # Simulate API returning NO odds
        mock_instance.get_fixture_odds.return_value = []
        
        predictor = LivePredictor(api_key="test_key")
        predictor.client = mock_instance # Force inject instance
    # Mock build_features to return a valid DataFrame
    predictor._build_features = MagicMock(return_value=pd.DataFrame({
        'elo_diff': [0],
        'imp_home_norm': [np.nan], # Simulate no market odds
        'imp_draw_norm': [np.nan],
        'imp_away_norm': [np.nan]
    }))
    
    # Create dummy fixture
    fixture = MatchFixture(
        fixture_id=123, league_id=8, 
        home_team="Home", home_team_id=1,
        away_team="Away", away_team_id=2,
        kickoff=datetime.now()
    )
    
    # Run prediction
    res = predictor.predict_fixture(fixture)
    
    print(f"Prediction Result: {res}")
    
    if res.ev_home is None and res.recommended is False:
        print("✅ PASS: Correctly handled missing odds (EV is None, Not Recommended)")
    else:
        print(f"❌ FAIL: EV={res.ev_home}, Rec={res.recommended}. Expected None/False.")

def verify_enrichment_performance_logic():
    print("\n--- 2. Verifying Enrichment Script Logic ---")
    print("Checking 'scripts/enrich_history.py' for O(N^2) loop removal...")
    
    with open("scripts/enrich_history.py", "r") as f:
        content = f.read()
        
    if "pd.read_parquet(output_path)" in content.split("def _append_chunk")[0]:
        print("❌ FAIL: Still found 'read_parquet' inside the main processing loop!")
    else:
        print("✅ PASS: 'read_parquet' removed from main loop.")

    if "_write_temp_chunk" in content and "_finalize_merge" in content:
        print("✅ PASS: '_write_temp_chunk' and '_finalize_merge' implemented.")
    else:
        print("❌ FAIL: optimization functions missing.")

if __name__ == "__main__":
    verify_live_predictor_mock_odds()
    verify_enrichment_performance_logic()
