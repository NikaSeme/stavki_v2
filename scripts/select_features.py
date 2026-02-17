
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
import sys

# Ensure stavki is in path
sys.path.append(str(Path.cwd()))

from stavki.pipelines.training import TrainingPipeline, TrainingConfig
from stavki.models.catboost import CatBoostModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def select_features():
    print("Loading data and pipeline...")
    data_path = Path("data/training_data.csv")
    if not data_path.exists():
        # Fallback to historical.csv if exists, else try full path
        data_path = Path("data/historical.csv")
        
    config = TrainingConfig(data_path=data_path)
    pipeline = TrainingPipeline(config=config)
    
    # Load and Split
    try:
        df = pipeline._load_data(config.data_path, None)
    except FileNotFoundError:
        print(f"Data not found at {config.data_path}")
        return

    # Use smaller subset for speed if large? No, use full train for accuracy.
    train_df, _, _ = pipeline._split_data(df)
    
    # Build Features
    print("Building features (this might take a while)...")
    # We need to capture X_train, y_train
    # _build_features returns X, y
    try:
        X_train, y_train = pipeline._build_features(train_df, fit_registry=True)
    except Exception as e:
        print(f"Feature building failed: {e}")
        return
    
    # Train CatBoost to get importance
    print(f"Training CatBoost on {len(X_train)} samples with {X_train.shape[1]} features...")
    model = CatBoostModel()
    
    # Prepare data for CatBoostModel wrapper fits
    # It expects DataFrame with target column, or X, y depending on wrapper?
    # training.py line 688: train_data = X_train.copy(); train_data["target"] = y_train
    
    train_data = X_train.copy()
    train_data["target"] = y_train
    
    try:
        # Use eval_ratio for internal validation
        model.fit(train_data, eval_ratio=0.15)
        
        # Get importance from underlying CatBoost model object
        # The wrapper stores `self.model` which is CatBoostClassifier
        cb = model.model
        importance = cb.get_feature_importance(type="FeatureImportance")
        feature_names = cb.feature_names_
        
        # Sort features
        features_sorted = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
        
        print("-" * 50)
        print(f"{'Feature':<35} | {'Importance':<10}")
        print("-" * 50)
        
        selected = []
        
        for f, imp in features_sorted:
            if imp > 0:
                print(f"{f:<35} | {imp:.4f}")
                # Threshold: Importance > 0.5 (relative score 0-100 usually in CatBoost)
                # Or use top N
                if imp >= 0.5:
                    selected.append(f)
                    
        print("-" * 50)
        
        # Fallback if too few
        if len(selected) < 10:
            print("Warning: Strict threshold selected too few features. Taking top 20.")
            selected = [f for f, _ in features_sorted[:20]]
            
        print(f"Selected {len(selected)} features.")
        
        # Save
        output_path = Path("config/selected_features.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(selected, f, indent=2)
            
        print(f"Saved selected features to {output_path}")
        
    except Exception as e:
        print(f"Selection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    select_features()
