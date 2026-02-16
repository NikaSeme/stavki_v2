"""
Training Pipeline
=================

Pipeline for training and optimizing models:
1. Load historical data
2. Build training features
3. Train individual models
4. Optimize ensemble weights
5. Calibrate thresholds
6. Save optimized config

Usage:
    pipeline = TrainingPipeline()
    results = pipeline.run(data_path="data/historical.csv")
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import click

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Data
    data_path: Path = field(default_factory=lambda: Path("data/historical.csv"))
    test_size: float = 0.20
    val_size: float = 0.10
    
    # Models to train
    models: List[str] = field(default_factory=lambda: ["poisson", "catboost", "neural"])
    
    # Training params
    epochs: int = 100
    early_stopping: int = 10
    
    # Optimization
    optimize_weights: bool = True
    optimize_thresholds: bool = True
    optimize_kelly: bool = True
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("models"))
    save_checkpoints: bool = True


@dataclass
class TrainingResult:
    """Results from training pipeline."""
    model_name: str
    accuracy: float
    log_loss: float
    roi_simulated: float
    training_time: float
    best_epoch: Optional[int] = None
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "log_loss": round(self.log_loss, 4),
            "roi_simulated": round(self.roi_simulated, 4),
            "training_time": round(self.training_time, 2),
            "best_epoch": self.best_epoch,
        }


class TrainingPipeline:
    """
    Pipeline for training and optimizing all models.
    
    Steps:
    1. Load and split data
    2. Build features
    3. Train models
    4. Evaluate on test set
    5. Optimize ensemble weights
    6. Calibrate strategy thresholds
    7. Save all artifacts
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()
        self.results: List[TrainingResult] = []
        
        # Will be populated during run
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.trained_models: Dict[str, Any] = {}  # Stores trained model objects
        self.optimal_weights: Dict[str, Dict[str, float]] = {}
        self.optimal_thresholds: Dict[str, float] = {}
        self.optimal_kelly: float = 0.25
    
    def run(
        self,
        data_path: Optional[Path] = None,
        data_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute full training pipeline.
        
        Args:
            data_path: Path to historical data CSV
            data_df: Pre-loaded DataFrame (optional)
        
        Returns:
            Dictionary with all training results and optimized configs
        """
        logger.info("=" * 60)
        logger.info("Starting Training Pipeline")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Step 1: Load data
        logger.info("Step 1: Loading data...")
        df = self._load_data(data_path, data_df)
        logger.info(f"  → {len(df)} matches loaded")
        
        # Step 2: Split data (temporal)
        logger.info("Step 2: Splitting data temporally...")
        self.train_df, self.val_df, self.test_df = self._split_data(df)
        logger.info(f"  → Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # Step 3: Build features
        logger.info("Step 3: Building features...")
        X_train, y_train = self._build_features(self.train_df)
        X_val, y_val = self._build_features(self.val_df)
        X_test, y_test = self._build_features(self.test_df)
        logger.info(f"  → {X_train.shape[1]} features")
        
        # Step 4: Train models
        logger.info("Step 4: Training models...")
        model_results = self._train_models(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        # Step 5: Optimize ensemble weights
        if self.config.optimize_weights:
            logger.info("Step 5: Optimizing ensemble weights...")
            self.optimal_weights = self._optimize_weights(X_test, y_test)
        
        # Step 6: Optimize thresholds
        if self.config.optimize_thresholds:
            logger.info("Step 6: Optimizing thresholds...")
            self.optimal_thresholds = self._optimize_thresholds()
        
        # Step 7: Optimize Kelly fraction
        if self.config.optimize_kelly:
            logger.info("Step 7: Optimizing Kelly fraction...")
            self.optimal_kelly = self._optimize_kelly()
        
        # Step 8: Save all artifacts
        logger.info("Step 8: Saving artifacts...")
        self._save_artifacts()
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"Training complete in {elapsed:.1f}s")
        for r in self.results:
            logger.info(f"  {r.model_name}: acc={r.accuracy:.2%}, ROI={r.roi_simulated:+.2%}")
        logger.info("=" * 60)
        
        return {
            "model_results": [r.to_dict() for r in self.results],
            "optimal_weights": self.optimal_weights,
            "optimal_thresholds": self.optimal_thresholds,
            "optimal_kelly": self.optimal_kelly,
            "elapsed_seconds": elapsed,
        }
    
    def _load_data(
        self,
        data_path: Optional[Path],
        data_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Load historical data."""
        if data_df is not None:
            return data_df
        
        path = data_path or self.config.data_path
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        df = pd.read_csv(path)
        
        # Ensure required columns
        required = ["HomeTeam", "AwayTeam", "FTR"]  # Full Time Result
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def _split_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (no shuffling to avoid lookahead bias).
        """
        n = len(df)
        train_end = int(n * (1 - self.config.test_size - self.config.val_size))
        val_end = int(n * (1 - self.config.test_size))
        
        train = df.iloc[:train_end].copy()
        val = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:].copy()
        
        return train, val, test
    
    def _build_features(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix and target."""
        # Map FTR to numeric
        ftr_map = {"H": 0, "D": 1, "A": 2}
        
        if "FTR" in df.columns:
            y = df["FTR"].map(ftr_map)
        elif "Result" in df.columns:
            y = df["Result"].map(ftr_map)
        else:
            y = pd.Series([0] * len(df))
            
        # Select numeric features
        feature_cols = [
            c for c in df.columns
            if c not in ["HomeTeam", "AwayTeam", "Date", "FTR", "Result", "Season", "League"]
            and df[c].dtype in [np.int64, np.float64, int, float]
        ]
        
        X = df[feature_cols].copy() if feature_cols else pd.DataFrame(index=df.index)
        X = X.fillna(0)
        
        # Drop rows where target is NaN (invalid FTR)
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask].astype(int)
        
        return X, y
    
    def _train_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Train all configured models and store them for optimization."""
        import time
        
        results = {}
        
        for model_name in self.config.models:
            logger.info(f"  Training {model_name}...")
            start = time.time()
            
            try:
                if model_name == "poisson":
                    result = self._train_poisson(X_train, y_train, X_test, y_test)
                elif model_name == "catboost":
                    result = self._train_catboost(X_train, y_train, X_val, y_val, X_test, y_test)
                elif model_name == "lightgbm":
                    result = self._train_lightgbm(X_train, y_train, X_val, y_val, X_test, y_test)
                elif model_name == "neural":
                    result = self._train_neural(X_train, y_train, X_val, y_val, X_test, y_test)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                result.training_time = time.time() - start
                self.results.append(result)
                results[model_name] = result
                
                logger.info(f"    → Accuracy: {result.accuracy:.2%}, ROI: {result.roi_simulated:+.2%}")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                click.echo(f"❌ Failed to train {model_name}: {e}")
        
        return results
    
    def _train_poisson(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingResult:
        """Train Poisson model (uses goals data)."""
        try:
            from stavki.models.poisson import DixonColesModel
            
            # Need train_df for goals
            if self.train_df is None:
                raise ValueError("Train data not available")
            
            model = DixonColesModel()
            model.fit(self.train_df)
            
            # Evaluate using bulk prediction
            preds = model.predict(self.test_df)
            
            # DixonColesModel.predict returns [1x2, OU, BTTS] for each match
            # We filter for 1X2 market to align with test_df rows
            from stavki.models.base import Market
            match_winner_preds = [p for p in preds if p.market == Market.MATCH_WINNER]
            
            correct = 0
            total = 0
            
            # Ensure alignment
            if len(match_winner_preds) != len(self.test_df):
                logger.warning(f"Poisson eval mismatch: {len(match_winner_preds)} preds vs {len(self.test_df)} actuals")
            
            for i, pred in enumerate(match_winner_preds):
                if i >= len(self.test_df):
                    break
                    
                if pred and pred.probabilities:
                    pred_outcome = max(pred.probabilities.items(), key=lambda x: x[1])[0]
                    actual_ftr = self.test_df.iloc[i].get("FTR")
                    
                    outcome_map = {"home": "H", "draw": "D", "away": "A"}
                    if outcome_map.get(pred_outcome) == actual_ftr:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            return TrainingResult(
                model_name="poisson",
                accuracy=accuracy,
                log_loss=0.0,
                roi_simulated=0.0,
                training_time=0.0,
            )
            
        except Exception as e:
            logger.warning(f"Poisson training failed: {e}")
            click.echo(f"❌ Poisson training failed: {e}")
            return TrainingResult("poisson", 0.33, 1.0, 0.0, 0.0)
    
    def _train_catboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingResult:
        """Train CatBoost model."""
        try:
            from stavki.models.catboost import CatBoostModel
            
            model = CatBoostModel()
            
            # CatBoostModel.fit expects a DataFrame with features AND target
            if self.train_df is None:
                 raise ValueError("Train data missing")
                 
            # NOTE: model.fit splits data internally (temporal split).
            # We pass self.train_df directly to respect the model's interface.
            
            model.fit(
                self.train_df, 
                eval_ratio=0.15 
            )
            
            # Predict uses X_test columns
            preds = model.predict(self.test_df)
            
            # Evaluate using standard logic since predict returns Predictions
            correct = 0
            total = 0
            for i, p in enumerate(preds):
                 if p.market == Market.MATCH_WINNER:
                     outcome = max(p.probabilities.items(), key=lambda x: x[1])[0]
                     actual = self.test_df.iloc[total].get("FTR")
                     
                     outcome_map = {"home": "H", "draw": "D", "away": "A"}
                     if outcome_map.get(outcome) == actual:
                         correct += 1
                     total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            # Store for optimization
            self.trained_models["catboost"] = model
            
            return TrainingResult(
                model_name="catboost",
                accuracy=accuracy,
                log_loss=0.0, 
                roi_simulated=0.0,
                training_time=0.0,
                feature_importance=model.get_feature_importance(),
            )
            
        except Exception as e:
            logger.warning(f"CatBoost training failed: {e}")
            click.echo(f"❌ CatBoost training failed: {e}")
            return TrainingResult("catboost", 0.33, 1.0, 0.0, 0.0)
    
    def _train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingResult:
        """Train LightGBM model."""
        try:
            from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
            
            model = LightGBMModel()
            
            if self.train_df is None:
                raise ValueError("Train data missing")
                
            model.fit(
                self.train_df,
                eval_ratio=0.15
            )
            
            # Predict
            preds = model.predict(self.test_df)
            
            correct = 0
            total = 0
            for i, p in enumerate(preds):
                 outcome = max(p.probabilities. items(), key=lambda x: x[1])[0]
                 actual = self.test_df.iloc[total].get("FTR")
                 
                 outcome_map = {"home": "H", "draw": "D", "away": "A"}
                 if outcome_map.get(outcome) == actual:
                     correct += 1
                 total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            # Store for optimization
            self.trained_models["lightgbm"] = model
            
            return TrainingResult(
                model_name="lightgbm",
                accuracy=accuracy,
                log_loss=0.0,
                roi_simulated=0.0,
                training_time=0.0,
            )
            
        except Exception as e:
            logger.warning(f"LightGBM training failed: {e}")
            click.echo(f"❌ LightGBM training failed: {e}")
            return TrainingResult("lightgbm", 0.33, 1.0, 0.0, 0.0)
    
    def _train_neural(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> TrainingResult:
        """Train Neural model."""
        try:
            from stavki.models.neural import MultiTaskModel
            
            model = MultiTaskModel(
                input_dim=X_train.shape[1],
                hidden_dims=[128, 64],
            )
            
            model.fit(
                X_train.values, y_train.values,
                val_data=(X_val.values, y_val.values),
                epochs=self.config.epochs,
                patience=self.config.early_stopping,
            )
            
            y_pred = model.predict_proba(X_test.values)
            y_pred_class = np.array(y_pred["1x2"]).argmax(axis=1)
            
            accuracy = (y_pred_class == y_test).mean()
            
            # Store for optimization
            self.trained_models["neural"] = model
            
            return TrainingResult(
                model_name="neural",
                accuracy=accuracy,
                log_loss=0.0,
                roi_simulated=0.0,
                training_time=0.0,
                best_epoch=model.best_epoch if hasattr(model, "best_epoch") else None,
            )
            
        except Exception as e:
            logger.warning(f"Neural training failed: {e}")
            click.echo(f"❌ Neural training failed: {e}")
            return TrainingResult("neural", 0.33, 1.0, 0.0, 0.0)
    
    def _compute_log_loss(
        self,
        y_pred: np.ndarray,
        y_true: pd.Series,
    ) -> float:
        """Compute log loss."""
        eps = 1e-10
        n_classes = y_pred.shape[1] if len(y_pred.shape) > 1 else 3
        
        loss = 0
        for i, true_class in enumerate(y_true):
            if true_class >= 0 and true_class < n_classes:
                prob = y_pred[i, int(true_class)] if len(y_pred.shape) > 1 else y_pred[i]
                loss -= np.log(max(prob, eps))
        
        return loss / len(y_true) if len(y_true) > 0 else 0
    
    def _optimize_weights(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Dict[str, float]]:
        """Optimize ensemble weights using trained models' predictions on test set."""
        try:
            from stavki.strategy import WeightOptimizer
            
            if len(self.trained_models) < 2:
                logger.info("  → Not enough trained models for weight optimization, using defaults")
                return {
                    "default": {name: 1.0 / len(self.trained_models) for name in self.trained_models}
                } if self.trained_models else {}
            
            # Collect probability predictions from each model on test set
            model_predictions = {}
            for name, model in self.trained_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_test.values)
                        if isinstance(proba, dict) and '1x2' in proba:
                            proba = np.array(proba['1x2'])
                        proba_df = pd.DataFrame(
                            proba, index=X_test.index,
                            columns=["H", "D", "A"][:proba.shape[1] if hasattr(proba, 'shape') and len(proba.shape) > 1 else 3]
                        )
                    elif hasattr(model, 'predict'):
                        preds = model.predict(X_test.values)
                        if isinstance(preds, np.ndarray) and len(preds.shape) == 2:
                            proba_df = pd.DataFrame(
                                preds, index=X_test.index,
                                columns=["H", "D", "A"][:preds.shape[1]]
                            )
                        else:
                            continue
                    else:
                        continue
                    model_predictions[name] = proba_df
                except Exception as e:
                    logger.warning(f"  → Predictions from {name} failed: {e}")
            
            if len(model_predictions) < 2:
                logger.info("  → Not enough valid predictions for optimization, using defaults")
                return {
                    "default": {name: 1.0 / len(model_predictions) for name in model_predictions}
                } if model_predictions else {}
            
            # Map y_test (0, 1, 2) back to column labels
            label_map = {0: "H", 1: "D", 2: "A"}
            actual_outcomes = y_test.map(label_map)
            
            # Create a minimal odds DataFrame (use test_df if it has odds)
            odds_data = pd.DataFrame(index=X_test.index)
            if self.test_df is not None:
                for col in self.test_df.columns:
                    if 'odds' in col.lower() or col in ['B365H', 'B365D', 'B365A', 'BWH', 'BWD', 'BWA']:
                        odds_data[col] = self.test_df[col].values[:len(X_test)]
            
            optimizer = WeightOptimizer(step_size=0.10)
            result = optimizer.optimize_ensemble_weights(
                model_predictions, actual_outcomes, odds_data, metric="accuracy"
            )
            
            logger.info(f"  → Optimized weights: {result.optimal_weights}")
            return {"default": result.optimal_weights}
            
        except Exception as e:
            logger.warning(f"Weight optimization failed: {e}")
            return {}
    
    def _optimize_thresholds(self) -> Dict[str, float]:
        """Optimize EV and edge thresholds using simulated bets on test data."""
        try:
            from stavki.strategy import ThresholdOptimizer
            
            # Build simulated bet records from test data
            historical_bets = self._build_simulated_bets()
            
            if len(historical_bets) < 30:
                logger.info("  → Not enough simulated bets for threshold optimization, using defaults")
                return {"min_ev": 0.03, "min_edge": 0.02}
            
            optimizer = ThresholdOptimizer()
            result = optimizer.optimize(historical_bets, min_bets=30)
            
            logger.info(f"  → Optimized thresholds: {result}")
            return result
            
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}")
            return {"min_ev": 0.03, "min_edge": 0.02}
    
    def _optimize_kelly(self) -> float:
        """Optimize Kelly fraction using simulated bets on test data."""
        try:
            from stavki.strategy import KellyOptimizer
            
            historical_bets = self._build_simulated_bets()
            
            if len(historical_bets) < 30:
                logger.info("  → Not enough simulated bets for Kelly optimization, using default")
                return 0.25
            
            optimizer = KellyOptimizer()
            best_fraction, results = optimizer.optimize(historical_bets)
            
            logger.info(f"  → Optimal Kelly fraction: {best_fraction}")
            return best_fraction
            
        except Exception as e:
            logger.warning(f"Kelly optimization failed: {e}")
            return 0.25
    
    def _build_simulated_bets(self) -> List[Dict]:
        """Build simulated bet records from test data for optimizer consumption."""
        bets = []
        
        if self.test_df is None or not self.trained_models:
            return bets
        
        # Try to get ensemble probabilities from first available model
        model = next(iter(self.trained_models.values()))
        
        try:
            X_test, y_test = self._build_features(self.test_df)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test.values)
                if isinstance(proba, dict) and '1x2' in proba:
                    proba = np.array(proba['1x2'])
            elif hasattr(model, 'predict'):
                proba = model.predict(X_test.values)
                if not isinstance(proba, np.ndarray) or len(proba.shape) != 2:
                    return bets
            else:
                return bets
            
            # Map predictions to bet records
            ftr_map = {"H": 0, "D": 1, "A": 2}
            odds_cols = {
                0: [c for c in self.test_df.columns if c in ['B365H', 'PSH', 'BWH', 'IWH']],
                1: [c for c in self.test_df.columns if c in ['B365D', 'PSD', 'BWD', 'IWD']],
                2: [c for c in self.test_df.columns if c in ['B365A', 'PSA', 'BWA', 'IWA']],
            }
            
            for i in range(len(proba)):
                if i >= len(y_test):
                    break
                
                best_outcome = int(np.argmax(proba[i]))
                model_prob = float(proba[i][best_outcome])
                actual_class = int(y_test.iloc[i]) if y_test.iloc[i] >= 0 else -1
                
                # Get odds for the predicted outcome
                odds_col_list = odds_cols.get(best_outcome, [])
                odds = 2.0  # fallback
                if odds_col_list and i < len(self.test_df):
                    for col in odds_col_list:
                        val = self.test_df.iloc[i].get(col)
                        if val and val > 1:
                            odds = float(val)
                            break
                
                ev = model_prob * odds - 1
                edge = model_prob - (1.0 / odds) if odds > 1 else 0
                
                bets.append({
                    "prob": model_prob,
                    "model_prob": model_prob,
                    "odds": odds,
                    "ev": ev,
                    "edge": edge,
                    "stake": 10.0,
                    "result": "win" if best_outcome == actual_class else "loss",
                })
        
        except Exception as e:
            logger.warning(f"Failed to build simulated bets: {e}")
        
        return bets
    
    def _save_artifacts(self):
        """Save all training artifacts."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "models": [r.to_dict() for r in self.results],
            "optimal_weights": self.optimal_weights,
            "optimal_thresholds": self.optimal_thresholds,
            "optimal_kelly": self.optimal_kelly,
        }
        
        with open(output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save league config
        league_config = {
            "default": {
                "policy": "BET",
                "weights": self.optimal_weights.get("default", {}),
                "min_ev": self.optimal_thresholds.get("min_ev", 0.03),
                "kelly_fraction": self.optimal_kelly,
            },
            "leagues": {},
        }
        
        with open(output_dir / "league_config.json", "w") as f:
            json.dump(league_config, f, indent=2)
        
        logger.info(f"Saved artifacts to {output_dir}")


def run_training_pipeline(
    data_path: str = "data/historical.csv",
    models: List[str] = None,
) -> Dict[str, Any]:
    """Convenience function to run training pipeline."""
    config = TrainingConfig(
        data_path=Path(data_path),
        models=models or ["poisson", "catboost"],
    )
    pipeline = TrainingPipeline(config=config)
    return pipeline.run()
