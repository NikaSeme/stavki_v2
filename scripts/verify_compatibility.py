
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stavki.prediction.live import LivePredictor
from stavki.data.loader import UnifiedDataLoader
from stavki.data.processors.normalize import normalize_team_name
from stavki.config import DATA_DIR

def test_compatibility():
    print("🕵️♂️ BOB'S EXTENDED AUDIT: Compatibility Check")
    print("="*60)
    
    # 1. Check Normalization Logic directly
    print("\n1. Testing Normalization Logic:")
    test_cases = {
        "Man Utd": "Man United",
        "Manchester United": "Man United",
        "Man City": "Man City", 
        "Spurs": "Tottenham",
        "Tottenham Hotspur": "Tottenham",
        " Wolves ": "Wolves",
        "Wolverhampton Wanderers": "Wolves"
    }
    
    for raw, expected in test_cases.items():
        got = normalize_team_name(raw)
        status = "✅" if got == expected else "❌"
        print(f"   '{raw}' -> '{got}' (Expected: '{expected}') {status}")
        if got != expected:
            print(f"   ⚠️ MISMATCH! Compatibility broken for {raw}")
            return # Fail fast

    # 2. Check LivePredictor Data Loading
    print("\n2. Testing LivePredictor Data Loading:")
    try:
        # Initialize predictor (mock API key)
        predictor = LivePredictor(api_key="verify_only")
        
        # Check if ELO ratings are loaded and keyed correctly
        print(f"   Loaded ELO ratings for {len(predictor.elo_ratings)} teams")
        
        # Check specific key existence
        target_team = "Man United"
        if target_team in predictor.elo_ratings:
            print(f"   ✅ '{target_team}' found in ELO ratings (Elo: {predictor.elo_ratings[target_team]})")
        else:
            print(f"   ❌ '{target_team}' NOT found in ELO ratings!")
            # Check if "Manchester United" exists instead (which would be mapped but not in keys if keys are wrong)
            if "Manchester United" in predictor.elo_ratings:
               print("   ⚠️ FOUND 'Manchester United' INSTEAD! Mismatch with canonical list.")
                
        # Check Form
        if target_team in predictor.team_form:
             print(f"   ✅ '{target_team}' found in Form data")
        else:
             print(f"   ❌ '{target_team}' NOT found in Form data")

    except Exception as e:
        print(f"   ❌ Error initializing LivePredictor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_compatibility()
