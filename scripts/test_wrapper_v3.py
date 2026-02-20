"""
Test script to verify the DeepInteractionWrapper V3.
Checks:
1. Model loading with new dimensions (referee_dim, h2h_dim)
2. H2H cache initialization from matches_silver
3. Prediction generation with and without live referee data
"""
import sys
import pandas as pd
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stavki.models.deep_interaction_wrapper import DeepInteractionWrapper

def test_wrapper():
    print("Initializing Wrapper...")
    model = DeepInteractionWrapper()
    
    print("\nLoading Checkpoint...")
    model.load_checkpoint()
    
    print(f"\nModel Loaded: {model.is_fitted}")
    print(f"Referees known: {model._num_referees}")
    print(f"H2H Cache Entries: {len(model._h2h_cache)}")
    
    if not model.is_fitted:
        print("❌ FAILED: Model not loaded")
        return
        
    # Get some valid IDs for testing
    fb = model.feature_builder
    ids = list(fb._team_latest_vectors.keys())[:2]
    h_id, a_id = ids[0], ids[1]
    
    # Test 1: Standard Prediction (missing live referee)
    print("\nTest 1: Standard Prediction (Unknown Referee = 0)")
    df1 = pd.DataFrame([{
        'HomeTeam': 'Team A',
        'AwayTeam': 'Team B',
        'home_team_id': h_id,
        'away_team_id': a_id,
        'Date': '2025-01-01',
        'league_id': 9
    }])
    
    pred1 = model.predict(df1)
    if pred1:
        p = pred1[0].probabilities
        print(f"✅ Success: H={p['home']:.4f}, D={p['draw']:.4f}, A={p['away']:.4f}")
    else:
        print("❌ FAILED")
        
    # Test 2: Prediction with Known Referee
    # Grab a real referee ID from the map
    ref_id = next(iter(model._referee_map.keys()))
    
    print(f"\nTest 2: Prediction with Known Referee (ID: {ref_id})")
    df2 = pd.DataFrame([{
        'HomeTeam': 'Team A',
        'AwayTeam': 'Team B',
        'home_team_id': h_id,
        'away_team_id': a_id,
        'Date': '2025-01-01',
        'league_id': 9,
        'referee_id': ref_id
    }])
    
    pred2 = model.predict(df2)
    if pred2:
        p2 = pred2[0].probabilities
        print(f"✅ Success: H={p2['home']:.4f}, D={p2['draw']:.4f}, A={p2['away']:.4f}")
        
        # Verify predictions are slightly different due to referee embedding
        p1 = pred1[0].probabilities
        diff = abs(p1['home'] - p2['home'])
        print(f"Difference in Home Prob due to Referee: {diff:.6f}")
    else:
        print("❌ FAILED")

if __name__ == "__main__":
    test_wrapper()
