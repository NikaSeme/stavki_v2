
import sys
import os
import time
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from stavki.features.registry import FeatureRegistry
from stavki.models.catboost.catboost_model import CatBoostModel
from stavki.pipelines.daily import DailyPipeline

def verify_optimizations():
    print("--- Verifying Optimizations ---")
    
    # 1. Feature Registry & CatBoost Integration
    print("\n[Test 1] CatBoost Feature Detection")
    try:
        registry = FeatureRegistry()
        all_features = registry.get_all_feature_names()
        print(f"  Registry reports {len(all_features)} features.")
        
        # Initialize model without explicit features
        model = CatBoostModel()
        
        # Create dummy df with subset of features to simulate data loading
        # Need enough rows for split
        dummy_data = {
            "Date": [datetime.now()] * 20,
            "HomeTeam": ["TeamA"] * 20,
            "AwayTeam": ["TeamB"] * 20,
            "target": [0, 1, 2, 0, 1] * 4,
            "league": ["L1"] * 20,
        }
        # Add some registry features
        for f in all_features[:10]:
            dummy_data[f] = [0.5] * 20
            
        dummy_df = pd.DataFrame(dummy_data)
        
        print("  Calling fit() to trigger autodetection...")
        try:
            # fit might fail due to catboost checks on tiny data, but features should be set
            model.fit(dummy_df, verbose=False, early_stopping_rounds=None)
        except Exception as e:
            print(f"  (Fit failed as expected on dummy data: {e})")
        
        if model.features and len(model.features) > 0:
             print(f"  PASS: CatBoost auto-detected {len(model.features)} features from registry.")
             print(f"  (Detected: {model.features})")
        else:
             print("  FAIL: CatBoost features list is empty.")
             
    except Exception as e:
        print(f"  FAIL: {e}")

    # 2. Data Loading Optimization
    print("\n[Test 2] History Loading Performance")
    pipeline = DailyPipeline()
    
    start = time.time()
    df = pipeline._load_history()
    elapsed = time.time() - start
    
    if not df.empty:
        print(f"  Loaded {len(df)} rows in {elapsed:.4f}s")
        if "HomeTeam" in df.columns:
            print(f"  HomeTeam dtype: {df['HomeTeam'].dtype}")
        print("  PASS: Data loaded successfully.")
    else:
        print("  WARN: No history data found (expected if no files exist).")

if __name__ == "__main__":
    verify_optimizations()
