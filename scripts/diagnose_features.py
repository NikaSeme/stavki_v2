#!/usr/bin/env python3
"""Diagnostic: show what features each model actually sees at prediction time"""
import sys, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", stream=sys.stdout)
for n in ["httpx","urllib3","httpcore"]: logging.getLogger(n).setLevel(logging.WARNING)

import pandas as pd
import numpy as np

def main():
    import pickle

    # Inspect the CatBoost model
    print("=== CATBOOST MODEL DIAGNOSTIC ===")
    with open("models/catboost.pkl", "rb") as f:
        state = pickle.load(f)
    model_features = state.get("features", [])
    print(f"Model expects {len(model_features)} features: {model_features}")

    cat_features = state.get("cat_features", [])
    print(f"Categorical features: {cat_features}")

    # Check training data
    train = pd.read_csv("data/features_full.csv")
    print(f"\nTraining data: {len(train)} rows, {len(train.columns)} columns")

    # Check each feature: training stats
    print(f"\n{'Feature':40s} | {'Train Mean':>10s} | {'Train Std':>10s} | {'Non-zero%':>10s}")
    print("-" * 80)
    for f in model_features:
        if f in train.columns and train[f].dtype in [np.float64, np.int64, float, int]:
            vals = train[f].dropna()
            nz = (vals != 0).mean() * 100
            print(f"{f:40s} | {vals.mean():10.4f} | {vals.std():10.4f} | {nz:9.1f}%")
        elif f in train.columns:
            nuniq = train[f].nunique()
            print(f"{f:40s} | {'(cat)':>10s} | {nuniq:>10d} | {'unique':>10s}")
        else:
            print(f"{f:40s} | {'MISSING':>10s} | {'?':>10s} | {'?':>10s}")

    # Check the LightGBM model
    print("\n\n=== LIGHTGBM MODEL DIAGNOSTIC ===")
    with open("models/LightGBM_1X2.pkl", "rb") as f:
        lgb_state = pickle.load(f)
    lgb_features = lgb_state.get("features", [])
    print(f"LightGBM expects {len(lgb_features)} features")
    missing_in_train = [f for f in lgb_features if f not in train.columns]
    if missing_in_train:
        print(f"  Missing from training data: {missing_in_train[:10]}...")

    # Check ensemble weights
    import json
    with open("models/league_weights.json") as f:
        weights = json.load(f)
    print(f"\n=== ENSEMBLE WEIGHTS ===")
    for league, w in weights.items():
        print(f"  {league}: {w}")

if __name__ == "__main__":
    main()
