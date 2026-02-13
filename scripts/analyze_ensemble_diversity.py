#!/usr/bin/env python3
"""
Analyze ensemble model diversity and correlation.
Helps understand why auto-optimizer chose extreme weights.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_ensemble import load_data, build_features, temporal_split, train_poisson, train_catboost, train_lightgbm

def main():
    # Load data
    df = load_data(PROJECT_ROOT / "data/features_full.csv")
    X, y, _ = build_features(df)
    data = temporal_split(df, X, y, test_frac=0.20)
    
    print("Training models for diversity analysis...\n")
    
    # Train all models
    poisson_proba, _, _ = train_poisson(data)
    catboost_proba, _, _, _ = train_catboost(data)
    lightgbm_proba, _, _, _ = train_lightgbm(data)
    
    y_test = data["y_test"].values
    
    # Compute individual log losses
    eps = 1e-10
    ll_poisson = log_loss(y_test, np.clip(poisson_proba, eps, 1-eps))
    ll_catboost = log_loss(y_test, np.clip(catboost_proba, eps, 1-eps))
    ll_lightgbm = log_loss(y_test, np.clip(lightgbm_proba, eps, 1-eps))
    
    print("Individual model log loss:")
    print(f"  Poisson:  {ll_poisson:.4f}")
    print(f"  CatBoost: {ll_catboost:.4f}")
    print(f"  LightGBM: {ll_lightgbm:.4f}")
    
    # Correlation analysis
    print("\n" + "="*60)
    print("Prediction Correlation Analysis")
    print("="*60)
    
    # Convert probabilities to class predictions
    pred_p = poisson_proba.argmax(axis=1)
    pred_c = catboost_proba.argmax(axis=1)
    pred_l = lightgbm_proba.argmax(axis=1)
    
    # Agreement rates
    agree_pc = (pred_p == pred_c).mean()
    agree_pl = (pred_p == pred_l).mean()
    agree_cl = (pred_c == pred_l).mean()
    agree_all = ((pred_p == pred_c) & (pred_c == pred_l)).mean()
    
    print(f"\nPrediction agreement rates:")
    print(f"  Poisson-CatBoost:  {agree_pc:.1%}")
    print(f"  Poisson-LightGBM:  {agree_pl:.1%}")
    print(f"  CatBoost-LightGBM: {agree_cl:.1%}")
    print(f"  All 3 models:      {agree_all:.1%}")
    
    # Error diversity analysis
    print(f"\n{'='*60}")
    print("Error Diversity Analysis")
    print("="*60)
    
    # Where each model is correct
    correct_p = pred_p == y_test
    correct_c = pred_c == y_test
    correct_l = pred_l == y_test
    
    # Unique correct predictions
    unique_p = correct_p & ~correct_c & ~correct_l
    unique_c = correct_c & ~correct_p & ~correct_l
    unique_l = correct_l & ~correct_p & ~correct_c
    
    print(f"\nUnique correct predictions (only one model got it right):")
    print(f"  Poisson only:  {unique_p.sum():4d} ({unique_p.mean():.1%})")
    print(f"  CatBoost only: {unique_c.sum():4d} ({unique_c.mean():.1%})")
    print(f"  LightGBM only: {unique_l.sum():4d} ({unique_l.mean():.1%})")
    
    # All correct
    all_correct = correct_p & correct_c & correct_l
    print(f"  All 3 correct: {all_correct.sum():4d} ({all_correct.mean():.1%})")
    
    # All wrong
    all_wrong = ~correct_p & ~correct_c & ~correct_l
    print(f"  All 3 wrong:   {all_wrong.sum():4d} ({all_wrong.mean():.1%})")
    
    # Complementary analysis
    print(f"\n{'='*60}")
    print("Complementary Value Analysis")
    print("="*60)
    
    # Where CatBoost is wrong but LightGBM is right
    cb_wrong_lgb_right = ~correct_c & correct_l
    print(f"\nCatBoost wrong, LightGBM right: {cb_wrong_lgb_right.sum():4d} ({cb_wrong_lgb_right.mean():.1%})")
    
   # Where CatBoost is wrong but Poisson is right
    cb_wrong_poi_right = ~correct_c & correct_p
    print(f"CatBoost wrong, Poisson right:  {cb_wrong_poi_right.sum():4d} ({cb_wrong_poi_right.mean():.1%})")
    
    # Test various weight combinations
    print(f"\n{'='*60}")
    print("Testing Alternative Weight Combinations")
    print("="*60)
    
    weight_configs = [
        {"name": "Current (optimized)", "w": [0.00, 0.999, 0.001]},
        {"name": "Equal weights",         "w": [1/3, 1/3, 1/3]},
        {"name": "CatBoost only",         "w": [0.0, 1.0, 0.0]},
        {"name": "50-50 CB-LGB",          "w": [0.0, 0.5, 0.5]},
        {"name": "70-20-10",              "w": [0.1, 0.7, 0.2]},
        {"name": "Min 10% per model",     "w": [0.1, 0.8, 0.1]},
    ]
    
    print(f"\n{'Config':<25} {'Log Loss':>10} {'Accuracy':>10}")
    print("-" * 60)
    
    for config in weight_configs:
        w = config["w"]
        ens = w[0] * poisson_proba + w[1] * catboost_proba + w[2] * lightgbm_proba
        ens = np.clip(ens, eps, 1-eps)
        ens = ens / ens.sum(axis=1, keepdims=True)
        
        ll = log_loss(y_test, ens)
        acc = (ens.argmax(axis=1) == y_test).mean()
        
        print(f"{config['name']:<25} {ll:>10.4f} {acc:>9.1%}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
