
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import json

# Ensure stavki is in path
sys.path.append(str(Path.cwd()))

from stavki.pipelines.training import TrainingPipeline, TrainingConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def optimize_roi():
    print("Loading pipeline and data...")
    config = TrainingConfig(data_path=Path("data/training_data.csv"))
    # Use feature selection for optimal performance
    config.use_feature_selection = True
    
    pipeline = TrainingPipeline(config=config)
    
    try:
        df = pipeline._load_data(config.data_path, None)
    except FileNotFoundError:
        print(f"Data not found at {config.data_path}")
        return

    # Split: Train/Val/Test
    # We optimize thresholds on VAL set to avoid overfitting Test set
    print("Splitting data...")
    train_df, val_df, test_df = pipeline._split_data(df)
    
    print("Fitting features...")
    # Fit on train
    X_train, y_train = pipeline._build_features(train_df, fit_registry=True)
    # Transform Val
    X_val, y_val = pipeline._build_features(val_df, fit_registry=False)
    
    # Apply selection
    if config.use_feature_selection:
        import json
        selected_path = Path("config/selected_features.json")
        if selected_path.exists():
            with open(selected_path) as f:
                selected = json.load(f)
            valid_selected = [c for c in selected if c in X_train.columns]
            if valid_selected:
                X_train = X_train[valid_selected]
                X_val = X_val[valid_selected]
    
    # Train CatBoost (our best model according to calibration)
    print("Training CatBoost on Train set...")
    from stavki.models.catboost import CatBoostModel
    model = CatBoostModel()
    train_data = X_train.copy()
    train_data["target"] = y_train
    model.fit(train_data)
    
    # Predict on Val
    print("Predicting on Validation set...")
    preds = model.predict(X_val)
    
    # Optimization Loop
    # Grid search over:
    # - Probability Threshold (min prob to bet)
    # - EV Threshold (min expected value)
    
    prob_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6]
    ev_thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10]
    
    best_roi = -1.0
    best_params = {}
    best_metrics = {}
    
    print("\nROI Optimization Grid Search:")
    print(f"{'Prob Thr':<10} | {'EV Thr':<10} | {'Bets':<6} | {'ROI':<8} | {'Profit':<8}")
    print("-" * 60)
    
    # Odds columns in val_df
    # We need to map val_df rows to preds
    # X_val has same index as val_df? Yes, if built correctly.
    # TrainingPipeline._build_features aligns them.
    
    # Let's align indices just in case
    val_df_aligned = val_df.loc[X_val.index]
    
    for p_thr in prob_thresholds:
        for ev_thr in ev_thresholds:
            
            total_stake = 0
            total_profit = 0
            bets_count = 0
            wins = 0
            
            for i, p in enumerate(preds):
                if not hasattr(p, "probabilities"): 
                    continue
                    
                # Identify best outcome
                probs = p.probabilities
                best_outcome = max(probs, key=probs.get)
                model_prob = probs[best_outcome]
                
                # Filter by Prob Threshold
                if model_prob < p_thr:
                    continue
                
                # Get Odds
                # Map outcome to columns
                # "home" -> B365H, PSH...
                # "draw" -> B365D...
                # "away" -> B365A...
                
                outcome_map = {"home": 0, "draw": 1, "away": 2}
                outcome_idx = outcome_map.get(best_outcome)
                
                odds_cols = {
                    "home": ["B365H", "PSH", "BWH"],
                    "draw": ["B365D", "PSD", "BWD"],
                    "away": ["B365A", "PSA", "BWA"],
                }
                
                row = val_df_aligned.iloc[i]
                odds = 0.0
                for col in odds_cols.get(best_outcome, []):
                    val = row.get(col)
                    if pd.notna(val) and val > 1:
                        odds = float(val)
                        break
                
                if odds <= 1:
                    continue
                    
                # Calculate EV
                ev = (model_prob * odds) - 1
                
                # Filter by EV Threshold
                if ev < ev_thr:
                    continue
                    
                # Place Bet
                stake = 10 # Flat stake for optimization
                bets_count += 1
                total_stake += stake
                
                # Check Result
                actual_res = row.get("FTR") # H, D, A
                ftr_map = {"H": "home", "D": "draw", "A": "away"}
                
                if ftr_map.get(actual_res) == best_outcome:
                    profit = stake * (odds - 1)
                    wins += 1
                else:
                    profit = -stake
                
                total_profit += profit
            
            roi = total_profit / total_stake if total_stake > 0 else 0.0
            
            # Print if valid sample
            if bets_count > 50:
                print(f"{p_thr:<10} | {ev_thr:<10} | {bets_count:<6} | {roi:+.2%} | {total_profit:.1f}")
                
                if roi > best_roi:
                    best_roi = roi
                    best_params = {"min_prob": p_thr, "min_ev": ev_thr}
                    best_metrics = {"bets": bets_count, "roi": roi, "profit": total_profit}
                    
    print("-" * 60)
    print(f"Best ROI: {best_roi:.2%}")
    print(f"Optimal Params: {best_params}")
    print(f"Metrics: {best_metrics}")
    
    # Save to config
    output_path = Path("config/betting_thresholds.json")
    with open(output_path, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"Saved optimal thresholds to {output_path}")

if __name__ == "__main__":
    optimize_roi()
