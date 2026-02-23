#!/usr/bin/env python3
"""
Online Learning Pipeline (Continual Retraining)
==============================================

This script executes Phase 17 of the STAVKI development lifecycle. 
It loads the freshly mapped `features_full.csv`, extracts the chronological batch of 
matches from the past `N` days, and incrementally fits the master Ensemble models 
using partial-fit and small learning rate configurations to prevent catastrophic forgetting.

Usage:
    python scripts/online_learning.py
    python scripts/online_learning.py --days-back 3
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.config import get_config
from stavki.models.ensemble.predictor import EnsemblePredictor
from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
from stavki.models.catboost.catboost_model import CatBoostModel
from stavki.models.gradient_boost.btts_model import BTTSModel
from stavki.models.deep_interaction_wrapper import DeepInteractionWrapper
from stavki.pipelines.daily import DailyPipeline
from stavki.features.registry import FeatureRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ContinualLearning")


def main():
    parser = argparse.ArgumentParser(description="Incrementally Retrain Models on Recent Match Vectors")
    parser.add_argument("--days-back", type=int, default=3, help="Number of days to isolate for incremental retraining")
    args = parser.parse_args()

    # Calculate cutoff
    cutoff_date = datetime.now() - timedelta(days=args.days_back)
    
    logger.info("=" * 60)
    logger.info(f"Initializing Continual Online Learning Engine")
    logger.info(f"Targeting active match features since {cutoff_date.strftime('%Y-%m-%d')}")
    logger.info("=" * 60)
    
    # 1. Load the Data
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        sys.exit(1)
        
    logger.info(f"Loading Master Table: {data_path.relative_to(PROJECT_ROOT)}")
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        sys.exit(1)
        
    if 'Date' not in df.columns:
        logger.error("Column 'Date' missing from dataset.")
        sys.exit(1)
        
    # Convert dates and slice
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    batch_df = df[df['Date'] >= cutoff_date].copy()
    
    if batch_df.empty:
        logger.warning(f"No completed matches found in `features_full.csv` since {cutoff_date.strftime('%Y-%m-%d')}.")
        logger.warning("Continual Learning skipped.")
        return
        
    logger.info(f"Isolated {len(batch_df)} ground-truth fixtures. Synthesizing ML Tensors...")
    
    # Run through the unified feature registry to dynamically generate elo, xG mappings, momentum etc.
    pipe = DailyPipeline()
    matches = pipe._df_to_matches(batch_df, is_historical=True)
    registry = FeatureRegistry(training_mode=False)
    
    df_features = registry.transform_historical(matches)
    batch_tensor = pipe._map_features_to_model_inputs(df_features)
    
    # Explicitly attach target logic for training
    batch_tensor['FTHG'] = batch_df['FTHG'].values
    batch_tensor['FTAG'] = batch_df['FTAG'].values
    batch_tensor['HomeTeam'] = batch_df['HomeTeam'].values
    batch_tensor['AwayTeam'] = batch_df['AwayTeam'].values
    batch_tensor['League'] = batch_df['League'].values
    batch_tensor['Date'] = batch_df['Date'].values
    
    # Outcome
    batch_tensor['Outcome'] = np.where(batch_tensor['FTHG'] > batch_tensor['FTAG'], 1, 
                                 np.where(batch_tensor['FTHG'] < batch_tensor['FTAG'], 2, 0))
    # BTTS
    batch_tensor['BTTS'] = np.where((batch_tensor['FTHG'] > 0) & (batch_tensor['FTAG'] > 0), 1, 0)
    
    # Define Core Model Save Paths
    model_dir = PROJECT_ROOT / "models"
    
    # 2. Retrain Boosting Trees (LightGBM)
    # LightGBM supports `init_model` during the .fit() method for natively appending trees
    lgb_path = model_dir / "LightGBM_1X2.pkl"
    if lgb_path.exists():
        logger.info("-> Updating LightGBM_1X2 Weights (Incremental Boosting)")
        try:
            lgb_model = LightGBMModel.load(lgb_path)
            if hasattr(lgb_model, 'model'):
                existing_booster = lgb_model.model
                # Fit natively
                lgb_model.fit(batch_tensor, incremental_base=existing_booster)
                lgb_model.save(lgb_path)
                logger.info(f"Successfully appended trees to {lgb_path.name}")
        except Exception as e:
            logger.error(f"Failed to update LightGBM: {e}")
            import traceback
            traceback.print_exc()
            
    # 3. Retrain Boosting Trees (CatBoost)
    # CatBoost also supports `init_model`, but standard configuration often requires explicit passing
    cb_path = model_dir / "catboost.pkl"
    if cb_path.exists():
        logger.info("-> Updating CatBoost Weights (Incremental Boosting)")
        try:
            cb_model = CatBoostModel.load(cb_path)
            if hasattr(cb_model, 'model'):
                existing_booster = cb_model.model
                cb_model.fit(batch_tensor, incremental_base=existing_booster)
                cb_model.save(cb_path)
                logger.info(f"Successfully appended trees to {cb_path.name}")
        except Exception as e:
            logger.error(f"Failed to update CatBoost: {e}")
            import traceback
            traceback.print_exc()

    # 4. Neural Network Exclusion Rule
    # The DeepInteraction v3 architecture consumes complex, 15-dimensional multi-head structures
    # generated by `DeepFeatureBuilder` (players, managers, H2H memory).
    # It cannot dynamically parse the flattened `batch_tensor` tabular format utilized by 
    # LightGBM and CatBoost. Neural incremental tuning is therefore delegated to the standard
    # full-retrain cycles executed via `train_deep_interaction.py`.
    
    logger.info("=" * 60)
    logger.info("Continual Learning Module Complete.")
    
if __name__ == "__main__":
    main()
