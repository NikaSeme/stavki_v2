import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from datetime import datetime

# Add project root
sys.path.insert(0, os.getcwd())

from stavki.prediction.live import LivePredictor, MatchFixture

def verify_formation_integration():
    print("\n--- Verifying Formation Feature Integration ---")
    
    # 1. Mock Data
    limit = 100
    mock_features = pd.DataFrame({
        'Date': [datetime.now()] * limit,
        'HomeTeam': ['TeamA'] * limit,
        'AwayTeam': ['TeamB'] * limit,
        'formation_home': ['3-4-3'] * limit, # Attacking (0.8)
        'formation_away': ['5-4-1'] * limit, # Defensive (0.2)
        'elo_home': [1500] * limit,
        'elo_away': [1500] * limit,
    })
    
    # 2. Patch dependencies
    with patch('stavki.prediction.live.SportMonksClient') as MockClient, \
         patch('pandas.read_parquet', return_value=mock_features), \
         patch('pandas.read_csv', return_value=mock_features):
        
        predictor = LivePredictor(api_key="test")
        
        # Check if builder is trained
        print(f"Builder fitted: {predictor.formation_builder._is_fitted}")
        
        team_a_fmts = predictor.formation_builder._team_formations.get('teama', [])
        print(f"TeamA formations: {len(team_a_fmts)} (Expected > 0)")
        
        if not team_a_fmts:
            print("❌ FAIL: FormationBuilder not populated from historical data!")
            return

        # 3. Test Prediction Feature Build
        fixture = MatchFixture(
             fixture_id=1, league_id=8,
             home_team='TeamA', home_team_id=1,
             away_team='TeamB', away_team_id=2,
             kickoff=datetime.now()
        )
        
        # Build features (pass mock odds)
        features_df = predictor._build_features(fixture, odds={})
        
        score_home = features_df['formation_score_home'].iloc[0]
        score_away = features_df['formation_score_away'].iloc[0]
        
        print(f"Home Score (TeamA 3-4-3): {score_home} (Expected ~0.8)")
        print(f"Away Score (TeamB 5-4-1): {score_away} (Expected ~0.2)")
        
        if abs(score_home - 0.8) < 0.1 and abs(score_away - 0.2) < 0.1:
            print("✅ PASS: Formation features correctly computed from historical preference!")
        else:
             print(f"❌ FAIL: Scores {score_home}/{score_away} do not match simple average logic.")

if __name__ == "__main__":
    verify_formation_integration()
