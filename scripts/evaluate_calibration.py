
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Ensure stavki is in path
sys.path.append(str(Path.cwd()))

from stavki.pipelines.training import TrainingPipeline, TrainingConfig
from stavki.models.base import Market

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_calibration():
    print("Loading pipeline and data...")
    config = TrainingConfig(data_path=Path("data/training_data.csv"))
    pipeline = TrainingPipeline(config=config)
    
    # Load and Split
    try:
        df = pipeline._load_data(config.data_path, None)
    except FileNotFoundError:
        print(f"Data not found at {config.data_path}")
        return

    _, _, test_df = pipeline._split_data(df)
    
    # We need to rebuild features for test_df
    # But to do that, we need a fitted registry.
    # We should really load the fitted registry or refit on train.
    # For accurate evaluation, we should refit on train.
    
    print("Re-fitting feature registry on training data...")
    train_df, _, _ = pipeline._split_data(df)
    
    # Fit registry
    X_train, y_train = pipeline._build_features(train_df, fit_registry=True)
    # Transform test
    X_test, y_test = pipeline._build_features(test_df, fit_registry=False)
    
    # Apply feature selection if configured (it is by default now)
    if config.use_feature_selection:
        import json
        selected_path = Path("config/selected_features.json")
        if selected_path.exists():
            with open(selected_path) as f:
                selected = json.load(f)
            # Filter columns
            valid_selected = [c for c in selected if c in X_test.columns]
            if valid_selected:
                X_test = X_test[valid_selected]
                # Also filter X_train for retraining if needed
                X_train = X_train[valid_selected]
    
    # Load trained models
    # We need to instantiate them and load weights?
    # Or just retrain them quickly? Retraining ensures consistency with current code/features.
    # Loading pickles might be risky if code changed (pickle is brittle).
    # Comparison: Retraining CatBoost/LightGBM on 16k rows is fast (<1 min).
    
    print("Retraining models for fresh evaluation...")
    from stavki.models.catboost import CatBoostModel
    from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
    from stavki.models.poisson import DixonColesModel
    
    models = {
        "CatBoost": CatBoostModel(),
        "LightGBM": LightGBMModel(),
        # "Poisson": DixonColesModel() # Poisson needs raw DF, handling separately if needed
    }
    
    # Train and Predict
    results = {}
    
    # Prepare CatBoost/LGBM data
    y_train_series = y_train
    
    # CatBoost
    print("Evaluating CatBoost...")
    train_data_cb = X_train.copy()
    train_data_cb["target"] = y_train_series
    models["CatBoost"].fit(train_data_cb)
    preds_cb = models["CatBoost"].predict(X_test)
    results["CatBoost"] = preds_cb
    
    # LightGBM
    print("Evaluating LightGBM...")
    reverse_ftr = {0: "H", 1: "D", 2: "A"}
    train_data_lgbm = X_train.copy()
    train_data_lgbm["target"] = y_train_series.map(reverse_ftr)
    models["LightGBM"].fit(train_data_lgbm)
    preds_lgbm = models["LightGBM"].predict(X_test)
    results["LightGBM"] = preds_lgbm
    
    # Calculate Metrics
    print("\n" + "="*60)
    print(f"{'Model':<15} | {'Outcome':<10} | {'Brier Score':<12} | {'ECE':<12}")
    print("="*60)
    
    for name, preds in results.items():
        # Extract probabilities for H, D, A
        probs_h = []
        probs_d = []
        probs_a = []
        
        for p in preds:
            if hasattr(p, "probabilities"):
                probs_h.append(p.probabilities.get("home", 0))
                probs_d.append(p.probabilities.get("draw", 0))
                probs_a.append(p.probabilities.get("away", 0))
                
        # Actuals
        # y_test is 0, 1, 2
        y_true_h = (y_test == 0).astype(int)
        y_true_d = (y_test == 1).astype(int)
        y_true_a = (y_test == 2).astype(int)
        
        # Brier Scores
        bs_h = brier_score_loss(y_true_h, probs_h)
        bs_d = brier_score_loss(y_true_d, probs_d)
        bs_a = brier_score_loss(y_true_a, probs_a)
        
        # Calibration Curves (ECE approx)
        # We can print print bin info for Home win
        print(f"{name:<15} | {'Home':<10} | {bs_h:.4f}       | {'-':<12}")
        print(f"{name:<15} | {'Draw':<10} | {bs_d:.4f}       | {'-':<12}")
        print(f"{name:<15} | {'Away':<10} | {bs_a:.4f}       | {'-':<12}")
        
        # Detailed calibration regarding Home Win (most important usually)
        prob_true, prob_pred = calibration_curve(y_true_h, probs_h, n_bins=10)
        
        print(f"\n{name} Calibration Table (Home Win):")
        print(f"{'Mean Pred':<10} | {'Fraction Pos':<12} | {'Count':<8}")
        # Note: calibration_curve doesn't return counts, strictly.
        # But we can assume uniform bins or similar. 
        # Actually printed points are sufficient.
        for p_pred, p_true in zip(prob_pred, prob_true):
             print(f"{p_pred:.4f}     | {p_true:.4f}       ")
        print("-" * 40)

if __name__ == "__main__":
    evaluate_calibration()
