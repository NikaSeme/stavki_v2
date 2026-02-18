
import sys
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Configure logging to capture everything
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("stavki.data.collectors").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Add project root
sys.path.append(os.getcwd())

from stavki.pipelines.daily import DailyPipeline, BetCandidate
from stavki.strategy.kelly import KellyStaker, EVResult
from stavki.features.registry import FeatureRegistry

def debug_system():
    print("--- DEEP SYSTEM AUDIT START ---")
    
    # 1. Initialize Pipeline (Force strict production mode)
    print("\n[Phase 1] Initialization")
    try:
        pipeline = DailyPipeline()
        pipeline._init_components()
        print("  Pipeline initialized.")
        
        # Check components
        print(f"  Staker Config: {pipeline._staker.config}")
        print(f"  Bankroll: {pipeline._staker.bankroll}")
        
    except Exception as e:
        print(f"  CRITICAL FAIL: Pipeline init failed: {e}")
        return

    # 2. Mock Data Injection (Simulate a "Good" Bet that fails)
    print("\n[Phase 2] Kelly Simulation (Mock Data)")
    
    # Scenario: Good EV, High Probability -> Should NOT be 0.00
    mock_bet = EVResult(
        match_id="debug_match_01",
        market="1x2",
        selection="home",
        model_prob=0.65,      # High confidence
        odds=2.0,             # Even money
        ev=0.30,              # Massive 30% EV
        edge_pct=0.30,
        implied_prob=0.50
    )
    
    print(f"  Input: Prob={mock_bet.model_prob}, Odds={mock_bet.odds}, EV={mock_bet.ev}")
    
    res = pipeline._staker.calculate_stake(mock_bet)
    print(f"  Result: Stake={res.stake_amount} ({res.stake_pct:.2%}), Reason={res.reason}")
    
    if res.stake_amount == 0.0:
        print("  FAIL: Good bet rejected! (See reason above)")
    else:
        print("  PASS: Good bet accepted.")

    # 3. Live Data Inspection (If available)
    print("\n[Phase 3] Live Data Inspection")
    try:
        # Load valid odds if possible
        odds_df = pipeline._fetch_odds()
        if not odds_df.empty:
            print(f"  Fetched {len(odds_df)} odds rows.")
            
            # Check columns
            print(f"  Odds Columns: {list(odds_df.columns)}")
            
            # Check for BTTS markets safely
            if 'market_key' in odds_df.columns:
                btts_markets = odds_df[odds_df['market_key'] == 'btts']
                print(f"  BTTS Markets found: {len(btts_markets)}")
                if len(btts_markets) == 0:
                    print("  WARN: No BTTS markets found in source data!")
            else:
                print("  WARN: 'market_key' column not found in odds_df.")
                
            # Run full pipeline on small subset
            matches_df = pipeline._extract_matches(odds_df.head(10)) # Limit to 10
            pipeline._enrich_matches(matches_df)
            features_df = pipeline._build_features(matches_df, odds_df)
            
            if features_df.empty:
                 print("  WARN: No features built.")
            else:
                 # Check for zero features
                 zeros = (features_df == 0).sum()
                 print(f"  Feature Zeros (Top 5): {zeros.sort_values(ascending=False).head(5).to_dict()}")
            
            # Check Model Predictions (Bias Analysis)
            # Access internal ensemble to get per-model breakdowns
            print("\n[Phase 3.1] Model Breakdown")
            
            # We need to run predictions manually for each model to see breakdown
            # because pipeline._get_predictions returns only ensembled result
            
            # Access cached ensemble
            if not getattr(pipeline, "ensemble", None):
                print("  Loading ensemble for inspection...")
                pipeline.ensemble = pipeline._load_ensemble()
                
            # Compatibility: Add legacy column aliases if missing (same as in daily.py)
            if "HomeTeam" not in features_df.columns and "home_team" in features_df.columns:
                features_df["HomeTeam"] = features_df["home_team"]
            if "AwayTeam" not in features_df.columns and "away_team" in features_df.columns:
                features_df["AwayTeam"] = features_df["away_team"]
            if "League" not in features_df.columns and "league" in features_df.columns:
                features_df["League"] = features_df["league"]
                
            ensemble = getattr(pipeline, "ensemble", None)
            if not ensemble or not ensemble.models:
                print("    WARN: No models found in pipeline.ensemble")
                models_dict = {}
            else:
                models_dict = ensemble.models
            
            for name, model in models_dict.items():
                print(f"  Checking {name}...")
                try:
                    # Check if model supports any of the markets
                    supports_any = False
                    for mkt in [1, 2, 3]: # MatchWinner, OverUnder, BTTS (Enum values or similar)
                         # We'd strictly need Market enum, but let's just try predict
                         pass
                    
                    raw_preds = model.predict(features_df)
                    if raw_preds and len(raw_preds) > 0:
                        # Inspect first prediction for 1x2
                        p = raw_preds[0]
                        if "home" in p.probabilities:
                            print(f"    Sample: {p.probabilities}")
                            
                            # Calculate average draw prob for this model
                            draws = [rp.probabilities.get("draw", 0) for rp in raw_preds if "draw" in rp.probabilities]
                            avg_draw = sum(draws) / len(draws) if draws else 0
                            print(f"    Avg Draw Prob: {avg_draw:.3f}")
                            
                            if avg_draw > 0.45:
                                print(f"    !! SUSPECT !! {name} is heavily biased towards draws.")
                        else:
                             print(f"    Sample (Non-1x2): {p.probabilities}")
                    else:
                        print(f"    WARN: {name} returned 0 predictions.")
                        
                except Exception as e:
                    print(f"    Error: {e}")
                    import traceback
                    traceback.print_exc()

            # [Phase 3.3] Specific Diagnosis
            print("\n[Phase 3.3] Specific Diagnosis")
            
            # Check HomeTeam column for NeuralMultiTask
            if "HomeTeam" in features_df.columns:
                print("  PASS: 'HomeTeam' is in features_df.")
            else:
                print("  FAIL: 'HomeTeam' missing from features_df columns:", list(features_df.columns))
                # Check aliases
                if "home_team" in features_df.columns:
                     print("  (But 'home_team' exists)")

            # Check DixonColes Loading
            from pathlib import Path
            from stavki.models.base import BaseModel
            
            dc_path = Path("models/DixonColes.pkl")
            if dc_path.exists():
                print(f"  Attempting to load {dc_path} manually...")
                try:
                    dc = BaseModel.load(dc_path)
                    print(f"  PASS: Loaded {dc.name} successfully.")
                except Exception as e:
                    print(f"  FAIL: Could not load DixonColes: {e}")
            else:
                print("  WARN: models/DixonColes.pkl not found.")

            preds = pipeline._get_predictions(matches_df, features_df)
            
            # Analyze distribution
            print("\n[Phase 3.4] Ensemble Distribution")
            home_probs = []
            draw_probs = []
            away_probs = []
            
            for match_id, markets in preds.items():
                if "1x2" in markets:
                    p = markets["1x2"]
                    home_probs.append(p["home"])
                    draw_probs.append(p["draw"])
                    away_probs.append(p["away"])
            
            if home_probs:
                print(f"  Avg Home Prob: {np.mean(home_probs):.3f}")
                print(f"  Avg Draw Prob: {np.mean(draw_probs):.3f}")
                print(f"  Avg Away Prob: {np.mean(away_probs):.3f}")
                
                if np.mean(draw_probs) > 0.40:
                    print("  WARN: Draw probability seems suspiciously high (>40%)")

    except Exception as e:
        print(f"  Live inspection failed: {e}")

if __name__ == "__main__":
    debug_system()
