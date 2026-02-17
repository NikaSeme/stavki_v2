"""
Model Trainer - Unified Training Orchestrator
==============================================

Handles:
- Data loading and preprocessing
- Sequential training of all models
- Ensemble weight optimization
- Calibration fitting
- Model persistence
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import json
import logging

from ..base import BaseModel, Market
from ..poisson import DixonColesModel
from ..gradient_boost import LightGBMModel, BTTSModel
from ..neural import MultiTaskModel, GoalsRegressor
from ..ensemble import EnsemblePredictor, EnsembleCalibrator, MarketAdjuster

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified trainer for all STAVKI models.
    
    Workflow:
    1. Load and validate data
    2. Temporal split (train/calibration/test)
    3. Train base models
    4. Fit calibration
    5. Optimize ensemble weights
    6. Save all models
    """
    
    def __init__(
        self,
        models_dir: Path,
        train_ratio: float = 0.70,
        cal_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Args:
            models_dir: Directory to save trained models
            train_ratio: Fraction of data for training
            cal_ratio: Fraction for calibration
            test_ratio: Fraction for testing
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_ratio = train_ratio
        self.cal_ratio = cal_ratio
        self.test_ratio = test_ratio
        
        # Initialize models
        self.models: Dict[str, BaseModel] = {}
        self.ensemble: Optional[EnsemblePredictor] = None
        self.calibrator: Optional[EnsembleCalibrator] = None
        
        # Training metadata
        self.training_log: List[Dict[str, Any]] = []
    
    def add_default_models(self):
        """Add default model configuration."""
        self.models = {
            "DixonColes": DixonColesModel(),
            "LightGBM_1X2": LightGBMModel(),
            "LightGBM_BTTS": BTTSModel(),
            "NeuralMultiTask": MultiTaskModel(),
            "GoalsRegressor": GoalsRegressor(),
        }
        
        # Market adjuster (doesn't need training)
        self.models["MarketAdjuster"] = MarketAdjuster()
        
        logger.info(f"Added {len(self.models)} default models")
    
    def add_model(self, model: BaseModel):
        """Add a custom model."""
        self.models[model.name] = model
    
    def train_all(
        self,
        data: pd.DataFrame,
        optimize_weights: bool = True,
        leagues: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train all models on provided data.
        
        Args:
            data: Full dataset with features and outcomes
            optimize_weights: Whether to optimize ensemble weights
            leagues: Optional list of leagues to optimize weights for
        
        Returns:
            Dict with training metrics for each model
        """
        if not self.models:
            self.add_default_models()
        
        # Sort by date
        df = data.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        n = len(df)
        train_end = int(n * self.train_ratio)
        cal_end = int(n * (self.train_ratio + self.cal_ratio))
        
        train_df = df.iloc[:train_end]
        cal_df = df.iloc[train_end:cal_end]
        test_df = df.iloc[cal_end:]
        
        logger.info(f"Data split: train={len(train_df)}, cal={len(cal_df)}, test={len(test_df)}")
        
        results = {
            "data_stats": {
                "total": n,
                "train": len(train_df),
                "calibration": len(cal_df),
                "test": len(test_df),
            },
            "models": {},
        }
        
        # Train each model
        for name, model in self.models.items():
            if name == "MarketAdjuster":
                continue  # Doesn't need training
            
            logger.info(f"Training {name}...")
            try:
                start_time = datetime.now()
                metrics = model.fit(train_df)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                results["models"][name] = {
                    "metrics": metrics,
                    "elapsed_seconds": elapsed,
                    "status": "success",
                }
                
                self.training_log.append({
                    "model": name,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success",
                    "metrics": metrics,
                })
                
                logger.info(f"  ✓ {name} trained in {elapsed:.1f}s")
                
                # Save immediately
                if self.models_dir:
                    model_path = self.models_dir / f"{name}.pkl"
                    model.save(model_path)

                
            except Exception as e:
                logger.error(f"  ✗ {name} failed: {e}")
                results["models"][name] = {
                    "status": "failed",
                    "error": str(e),
                }
        
        # Create ensemble
        self.ensemble = EnsemblePredictor(models=self.models)
        
        # Fit calibration
        logger.info("Fitting calibration...")
        self.calibrator = EnsembleCalibrator(method="isotonic")
        
        # Get predictions on calibration set
        cal_predictions = []
        for name, model in self.models.items():
            if model.is_fitted:
                try:
                    preds = model.predict(cal_df)
                    cal_predictions.extend(preds)
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {name}: {e}")
        
        # Build actuals dict
        actuals = {}
        for idx, row in cal_df.iterrows():
            from stavki.utils import generate_match_id
            match_id = row.get("match_id", generate_match_id(
                row.get('HomeTeam', ''), 
                row.get('AwayTeam', ''), 
                row.get('Date')
            ))
            
            # 1X2 actual
            if row["FTHG"] > row["FTAG"]:
                actuals[match_id] = "home"
            elif row["FTHG"] < row["FTAG"]:
                actuals[match_id] = "away"
            else:
                actuals[match_id] = "draw"
        
        self.calibrator.fit(cal_predictions, actuals)
        
        # Optimize ensemble weights
        if optimize_weights:
            logger.info("Optimizing ensemble weights...")
            
            markets = [Market.MATCH_WINNER, Market.OVER_UNDER, Market.BTTS]
            
            for market in markets:
                try:
                    weights = self.ensemble.optimize_weights(cal_df, market)
                    results[f"weights_{market.value}"] = weights
                except Exception as e:
                    logger.warning(f"Weight optimization failed for {market.value}: {e}")
            
            # Per-league optimization
            if leagues and "League" in cal_df.columns:
                for league in leagues:
                    league_data = cal_df[cal_df["League"] == league]
                    if len(league_data) < 30:
                        continue
                    
                    for market in markets:
                        try:
                            weights = self.ensemble.optimize_weights(
                                league_data, market, league=league
                            )
                        except Exception as e:
                            logger.warning(f"Weight optimization failed for {league}/{market.value}: {e}")
        
        # Save models
        self._save_all_models()
        
        results["ensemble_ready"] = True
        return results
    
    def _save_all_models(self):
        """Save all trained models to disk."""
        for name, model in self.models.items():
            if model.is_fitted:
                path = self.models_dir / f"{name}.pkl"
                try:
                    model.save(path)
                    logger.info(f"Saved {name} to {path}")
                except Exception as e:
                    logger.error(f"Failed to save {name}: {e}")
        
        # Save ensemble weights
        if self.ensemble:
            weights_path = self.models_dir / "ensemble_weights.json"
            self.ensemble.save_weights(weights_path)
        
        # Save training log
        log_path = self.models_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(self.training_log, f, indent=2)
    
    def load_models(self) -> Dict[str, BaseModel]:
        """Load all models from disk."""
        model_classes: Dict[str, Type[BaseModel]] = {
            "DixonColes": DixonColesModel,
            "LightGBM_1X2": LightGBMModel,
            "LightGBM_BTTS": BTTSModel,
            "NeuralMultiTask": MultiTaskModel,
            "GoalsRegressor": GoalsRegressor,
            "MarketAdjuster": MarketAdjuster,
        }
        
        self.models = {}
        
        for name, cls in model_classes.items():
            path = self.models_dir / f"{name}.pkl"
            if path.exists():
                try:
                    self.models[name] = cls.load(path)
                    logger.info(f"Loaded {name}")
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")
        
        # Load ensemble weights
        weights_path = self.models_dir / "ensemble_weights.json"
        if weights_path.exists():
            self.ensemble = EnsemblePredictor(models=self.models)
            self.ensemble.load_weights(weights_path)
        
        return self.models
    
    def predict(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate ensemble predictions."""
        if not self.ensemble:
            self.ensemble = EnsemblePredictor(models=self.models)
        
        predictions = self.ensemble.predict(data)
        
        # Calibrate if available
        if self.calibrator and self.calibrator.is_fitted:
            predictions = self.calibrator.calibrate(predictions)
        
        # Convert to dicts
        return [p.to_dict() for p in predictions]
