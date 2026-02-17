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

from stavki.models.base import Market

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    # Data
    data_path: Path = field(default_factory=lambda: Path("data/historical.csv"))
    test_size: float = 0.20
    val_size: float = 0.10
    
    # Models to train
    models: List[str] = field(default_factory=lambda: ["poisson", "catboost", "lightgbm", "neural"])
    
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
    use_feature_selection: bool = True
    
    # Walk-Forward Validation
    walk_forward: bool = False
    wf_train_months: int = 12
    wf_test_months: int = 3
    wf_step_months: int = 1


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
        
        self.registry = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_test: Optional[pd.Series] = None
    
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
        
        # Step 2: Split or Walk-Forward
        if self.config.walk_forward:
            return self._run_walk_forward(df)
            
        logger.info("Step 2: Splitting data temporally...")
        self.train_df, self.val_df, self.test_df = self._split_data(df)
        logger.info(f"  → Train: {len(self.train_df)}, Val: {len(self.val_df)}, Test: {len(self.test_df)}")
        
        # Step 3: Build features
        logger.info("Step 3: Building features...")
        # Fit registry on training data only
        X_train, y_train = self._build_features(self.train_df, fit_registry=True)
        # Reuse registry for val/test
        X_val, y_val = self._build_features(self.val_df, fit_registry=False)
        X_test, y_test = self._build_features(self.test_df, fit_registry=False)
        self.X_test = X_test
        self.y_test = y_test
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
        
        df = pd.read_csv(path, low_memory=False)
        
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
    
    @staticmethod
    def _safe_int(val) -> Optional[int]:
        """Safely convert a value to int, returning None for NaN/missing."""
        if pd.isna(val):
            return None
        try:
            return int(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        """Safely convert a value to float, returning None for NaN/missing."""
        if pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _compute_odds_features(row) -> Dict[str, float]:
        """Compute multi-bookmaker odds features from a CSV row.

        Returns implied probabilities (avg, spread) and closing line
        movement — all high-signal features for betting models.
        """
        # Bookmaker odds columns (H/D/A triplets)
        bookmakers = {
            "B365": ("B365H", "B365D", "B365A"),
            "BW":   ("BWH",   "BWD",   "BWA"),
            "IW":   ("IWH",   "IWD",   "IWA"),
            "PS":   ("PSH",   "PSD",   "PSA"),
            "WH":   ("WHH",   "WHD",   "WHA"),
            "VC":   ("VCH",   "VCD",   "VCA"),
        }

        implied_home, implied_draw, implied_away = [], [], []

        for _bk, (h_col, d_col, a_col) in bookmakers.items():
            h_odds = row.get(h_col)
            d_odds = row.get(d_col)
            a_odds = row.get(a_col)

            if pd.notna(h_odds) and pd.notna(d_odds) and pd.notna(a_odds):
                try:
                    h, d, a = float(h_odds), float(d_odds), float(a_odds)
                    if h > 1 and d > 1 and a > 1:
                        raw_sum = (1/h) + (1/d) + (1/a)
                        implied_home.append((1/h) / raw_sum)
                        implied_draw.append((1/d) / raw_sum)
                        implied_away.append((1/a) / raw_sum)
                except (ValueError, ZeroDivisionError):
                    pass

        features: Dict[str, float] = {}

        if implied_home:
            n = len(implied_home)
            features["avg_implied_home"] = round(sum(implied_home) / n, 4)
            features["avg_implied_draw"] = round(sum(implied_draw) / n, 4)
            features["avg_implied_away"] = round(sum(implied_away) / n, 4)
            features["odds_spread_home"] = round(max(implied_home) - min(implied_home), 4)
            features["odds_spread_draw"] = round(max(implied_draw) - min(implied_draw), 4)
            features["odds_spread_away"] = round(max(implied_away) - min(implied_away), 4)
        else:
            features["avg_implied_home"] = 0.0
            features["avg_implied_draw"] = 0.0
            features["avg_implied_away"] = 0.0
            features["odds_spread_home"] = 0.0
            features["odds_spread_draw"] = 0.0
            features["odds_spread_away"] = 0.0

        # Closing line movement (Pinnacle closing vs opening)
        for suffix, open_col, close_col in [
            ("home", "PSH", "PSCH"),
            ("draw", "PSD", "PSCD"),
            ("away", "PSA", "PSCA"),
        ]:
            o_val = row.get(open_col)
            c_val = row.get(close_col)
            if pd.notna(o_val) and pd.notna(c_val):
                try:
                    o, c = float(o_val), float(c_val)
                    if o > 1 and c > 1:
                        # Positive = odds shortened (sharp money backing)
                        features[f"closing_movement_{suffix}"] = round(
                            (1/c) - (1/o), 4
                        )
                    else:
                        features[f"closing_movement_{suffix}"] = 0.0
                except (ValueError, ZeroDivisionError):
                    features[f"closing_movement_{suffix}"] = 0.0
            else:
                features[f"closing_movement_{suffix}"] = 0.0

        return features

    def _build_features(
        self,
        df: pd.DataFrame,
        fit_registry: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Build feature matrix and target using FeatureRegistry."""
        try:
            from stavki.features.registry import FeatureRegistry
            from stavki.data.schemas import (
                Match, Team, League,
                MatchStats, MatchEnrichment, RefereeInfo,
            )
            from tqdm import tqdm
            
            # Initialize registry in training mode (skip API-only builders)
            if self.registry is None:
                self.registry = FeatureRegistry(training_mode=True)
            registry = self.registry
            
            # Convert DF to Match objects with full stats + enrichment
            matches = []
            logger.info("    Converting data to Match objects...")
            
            # Pre-parse dates to avoid repeated parsing in loop
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format='mixed')
            
            for idx, row in df.iterrows():
                try:
                    match_date = row["Date"]
                    if pd.isna(match_date):
                        continue
                    
                    # Map league string to Enum
                    league_str = str(row.get("League", "unknown")).lower()
                    league_map = {
                        "epl": League.EPL,
                        "premier_league": League.EPL,
                        "laliga": League.LA_LIGA,
                        "la_liga": League.LA_LIGA,
                        "bundesliga": League.BUNDESLIGA,
                        "seriea": League.SERIE_A,
                        "serie_a": League.SERIE_A,
                        "ligue1": League.LIGUE_1,
                        "ligue_1": League.LIGUE_1,
                        "championship": League.CHAMPIONSHIP,
                    }
                    league_enum = league_map.get(league_str, League.EPL)
                    
                    match_id = f"hist_{idx}"
                    
                    # --- Build MatchStats from CSV columns ---
                    stats = MatchStats(
                        match_id=match_id,
                        shots_home=self._safe_int(row.get("HS")),
                        shots_away=self._safe_int(row.get("AS")),
                        shots_on_target_home=self._safe_int(row.get("HST")),
                        shots_on_target_away=self._safe_int(row.get("AST")),
                        fouls_home=self._safe_int(row.get("HF")),
                        fouls_away=self._safe_int(row.get("AF")),
                        corners_home=self._safe_int(row.get("HC")),
                        corners_away=self._safe_int(row.get("AC")),
                        yellow_cards_home=self._safe_int(row.get("HY")),
                        yellow_cards_away=self._safe_int(row.get("AY")),
                        red_cards_home=self._safe_int(row.get("HR")),
                        red_cards_away=self._safe_int(row.get("AR")),
                    )
                    
                    # --- Build MatchEnrichment (referee) ---
                    referee = None
                    ref_name = row.get("Referee")
                    if pd.notna(ref_name) and str(ref_name).strip():
                        referee = RefereeInfo(name=str(ref_name).strip())
                    
                    enrichment = MatchEnrichment(referee=referee)
                    
                    m = Match(
                        id=match_id,
                        commence_time=match_date,
                        home_team=Team(name=str(row["HomeTeam"])),
                        away_team=Team(name=str(row["AwayTeam"])),
                        league=league_enum,
                        home_score=self._safe_int(row.get("FTHG")),
                        away_score=self._safe_int(row.get("FTAG")),
                        stats=stats,
                        enrichment=enrichment,
                    )
                    matches.append(m)
                except Exception as e:
                    if idx < 5:
                        logger.warning(f"Failed to convert row {idx}: {e}")
                    continue
            
            # Fit registry
            if fit_registry:
                logger.info(f"    Fitting FeatureRegistry on {len(matches)} matches...")
                self.registry.fit(matches)
            
            # Compute features for each match
            logger.info("    Computing features...")
            features_list = []
            valid_indices = []
            
            for i, m in enumerate(tqdm(matches, desc="Building features")):
                try:
                    feats = registry.compute(
                        home_team=m.home_team.name, 
                        away_team=m.away_team.name, 
                        as_of=m.commence_time,
                        match=m
                    )
                    
                    # Multi-bookmaker odds features
                    row = df.iloc[i]
                    odds_feats = self._compute_odds_features(row)
                    feats.update(odds_feats)
                    
                    features_list.append(feats)
                    valid_indices.append(df.index[i])
                    
                except Exception as e:
                    continue
            
            if not features_list:
                raise ValueError("No features computed")
                
            X = pd.DataFrame(features_list, index=valid_indices)
            X = X.fillna(0)
            
            # Align target
            ftr_map = {"H": 0, "D": 1, "A": 2}
            if "FTR" in df.columns:
                y_full = df["FTR"].map(ftr_map)
            elif "Result" in df.columns:
                y_full = df["Result"].map(ftr_map)
            else:
                y_full = pd.Series([0] * len(df), index=df.index)
                
            y = y_full.loc[valid_indices].astype(int)
            
            # Apply Feature Selection if configured
            if self.config.use_feature_selection:
                try:
                    import json
                    selected_path = Path("config/selected_features.json")
                    if selected_path.exists():
                        with open(selected_path) as f:
                            selected = json.load(f)
                        # Filter columns (keep only valid ones)
                        valid_selected = [c for c in selected if c in X.columns]
                        if valid_selected:
                            X = X[valid_selected]
                            logger.info(f"    Selected {len(valid_selected)} features from config")
                except Exception as e:
                    logger.warning(f"Feature selection failed: {e}")
            
            return X, y
            
        except Exception as e:
            logger.warning(f"FeatureRegistry build failed: {e}")
            logger.warning("Falling back to basic numeric features")
            import traceback
            traceback.print_exc()
            
            # Fallback (original logic)
            ftr_map = {"H": 0, "D": 1, "A": 2}
            if "FTR" in df.columns:
                y = df["FTR"].map(ftr_map)
            else:
                y = pd.Series([0] * len(df))
                
            feature_cols = [
                c for c in df.columns
                if c not in ["HomeTeam", "AwayTeam", "Date", "FTR", "Result", "Season", "League"]
                and df[c].dtype in [np.int64, np.float64, int, float]
            ]
            X = df[feature_cols].copy() if feature_cols else pd.DataFrame(index=df.index)
            X = X.fillna(0)
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
    
    def _simulate_roi(
        self,
        model_probs: List[Any],  # List of Prediction objects or dicts
        test_indices: pd.Index,
    ) -> float:
        """
        Simulate betting ROI on test set using actual odds.
        """
        try:
            from stavki.backtesting import BacktestEngine, BacktestConfig
            
            if self.test_df is None:
                return 0.0
                
            # Filter test_df to match test_indices
            # The test_df might be larger than X_test if we split earlier
            # But usually self.test_df corresponds to the test split
            # Let's align by index
            eval_df = self.test_df.loc[test_indices]
            
            if len(eval_df) < 10:
                return 0.0
                
            # Convert model_probs to dict format expected by BacktestEngine
            # Dict[match_idx, np.array([prob_h, prob_d, prob_a])]
            prob_dict = {}
            
            for i, idx in enumerate(test_indices):
                # Handle different prob formats
                p_obj = model_probs[i] if i < len(model_probs) else None
                
                if p_obj:
                    probs = None
                    if hasattr(p_obj, "probabilities"):
                        probs = p_obj.probabilities
                    elif isinstance(p_obj, dict):
                        probs = p_obj
                    
                    if probs:
                        prob_dict[idx] = np.array([
                            probs.get("home", 0.33),
                            probs.get("draw", 0.33),
                            probs.get("away", 0.34)
                        ])
            
            # Run simplified backtest
            config = BacktestConfig(
                min_ev=0.03,
                kelly_fraction=0.25,
                slippage=0.0  # No slippage for pure model signal check
            )
            engine = BacktestEngine(config)
            result = engine.run(eval_df, model_probs=prob_dict)
            
            return result.roi
            
        except Exception as e:
            logger.warning(f"ROI simulation failed: {e}")
            return 0.0

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
            
            # ROI Simulation
            roi = self._simulate_roi(match_winner_preds, self.test_df.index)
            
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
            
            # Register model for optimization steps
            self.trained_models["poisson"] = model
            
            return TrainingResult(
                model_name="poisson",
                accuracy=accuracy,
                log_loss=0.0,
                roi_simulated=roi,
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
            if X_train is None or y_train is None:
                 raise ValueError("Train data missing")
                 
            # Prepare training data with features AND target
            train_data = X_train.copy()
            train_data["target"] = y_train
            
            model.fit(
                train_data, 
                eval_ratio=0.15 
            )
            
            # Predict uses X_test columns (same feature space as training)
            preds = model.predict(X_test)
            
            # ROI Simulation
            roi = self._simulate_roi(preds, X_test.index)
            
            # Evaluate accuracy using y_test directly
            correct = 0
            total = 0
            for i, p in enumerate(preds):
                if p.market == Market.MATCH_WINNER:
                    outcome = max(p.probabilities.items(), key=lambda x: x[1])[0]
                    
                    # Map model output to numeric class
                    outcome_map = {"home": 0, "draw": 1, "away": 2}
                    pred_class = outcome_map.get(outcome, -1)
                    actual_class = int(y_test.iloc[i]) if i < len(y_test) else -1
                    
                    if pred_class == actual_class:
                        correct += 1
                    total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            # Compute log loss from predicted probabilities
            from sklearn.metrics import log_loss as sk_log_loss
            prob_matrix = []
            for p in preds:
                if p.market == Market.MATCH_WINNER and p.probabilities:
                    prob_matrix.append([
                        p.probabilities.get("home", 0.33),
                        p.probabilities.get("draw", 0.33),
                        p.probabilities.get("away", 0.34),
                    ])
            computed_log_loss = 0.0
            if prob_matrix and len(prob_matrix) == len(y_test):
                try:
                    computed_log_loss = sk_log_loss(
                        y_test.values, prob_matrix, labels=[0, 1, 2]
                    )
                except Exception:
                    pass
            
            # Store for optimization
            self.trained_models["catboost"] = model
            
            return TrainingResult(
                model_name="catboost",
                accuracy=accuracy,
                log_loss=computed_log_loss, 
                roi_simulated=roi,
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
            
            if X_train is None:
                raise ValueError("Train data missing")
                
            # Prepare training data with features AND target
            # LightGBM's predict() expects label_encoder classes to be 'H'/'D'/'A' strings
            # but y_train contains numeric 0/1/2 from the pipeline mapping
            reverse_ftr = {0: "H", 1: "D", 2: "A"}
            train_data = X_train.copy()
            train_data["target"] = y_train.map(reverse_ftr)
                
            model.fit(
                train_data,
                eval_ratio=0.15
            )
            
            # Predict using feature-engineered X_test (same feature space as training)
            preds = model.predict(X_test)
            
            # ROI Simulation
            roi = self._simulate_roi(preds, X_test.index)
            
            # Evaluate accuracy using y_test directly
            correct = 0
            total = 0
            for i, p in enumerate(preds):
                outcome = max(p.probabilities.items(), key=lambda x: x[1])[0]
                
                # Map model output to numeric class
                outcome_map = {"home": 0, "draw": 1, "away": 2}
                pred_class = outcome_map.get(outcome, -1)
                actual_class = int(y_test.iloc[i]) if i < len(y_test) else -1
                
                if pred_class == actual_class:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            
            # Compute log loss from predicted probabilities
            from sklearn.metrics import log_loss as sk_log_loss
            prob_matrix = []
            for p in preds:
                if p.probabilities:
                    prob_matrix.append([
                        p.probabilities.get("home", 0.33),
                        p.probabilities.get("draw", 0.33),
                        p.probabilities.get("away", 0.34),
                    ])
            computed_log_loss = 0.0
            if prob_matrix and len(prob_matrix) == len(y_test):
                try:
                    computed_log_loss = sk_log_loss(
                        y_test.values, prob_matrix, labels=[0, 1, 2]
                    )
                except Exception:
                    pass
            
            # Store for optimization
            self.trained_models["lightgbm"] = model
            
            return TrainingResult(
                model_name="lightgbm",
                accuracy=accuracy,
                log_loss=computed_log_loss,
                roi_simulated=roi,
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
            
            # ROI Simulation (need to convert neural probs to dict/objects)
            probs_list = []
            neural_probs = y_pred["1x2"]
            for i in range(len(neural_probs)):
                probs_list.append({
                    "home": neural_probs[i][0],
                    "draw": neural_probs[i][1],
                    "away": neural_probs[i][2],
                })
            
            roi = self._simulate_roi(probs_list, X_test.index)
            
            accuracy = (y_pred_class == y_test).mean()
            
            # Store for optimization
            self.trained_models["neural"] = model
            
            return TrainingResult(
                model_name="neural",
                accuracy=accuracy,
                log_loss=0.0,
                roi_simulated=roi,
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
                    if name == "poisson":
                        # Poisson needs raw DataFrame with team names
                        preds = model.predict(self.test_df)
                        
                        if isinstance(preds, list) and len(preds) > 0 and hasattr(preds[0], 'probabilities'):
                            rows = []
                            for p in preds:
                                if p.market == Market.MATCH_WINNER:
                                     probs = p.probabilities
                                     row = {
                                         "H": probs.get("home", 0.0),
                                         "D": probs.get("draw", 0.0),
                                         "A": probs.get("away", 0.0)
                                     }
                                     rows.append(row)
                            proba_df = pd.DataFrame(rows, index=X_test.index[:len(rows)])
                        else:
                            logger.warning(f"  → Unknown prediction format from {name}: {type(preds)}")
                            continue
                    else:
                        # CatBoost / LightGBM need feature-engineered X_test
                        preds = model.predict(self.X_test)
                        
                        if isinstance(preds, list) and len(preds) > 0 and hasattr(preds[0], 'probabilities'):
                            rows = []
                            for p in preds:
                                if p.market == Market.MATCH_WINNER:
                                    probs = p.probabilities
                                    row = {
                                        "H": probs.get("home", 0.0),
                                        "D": probs.get("draw", 0.0),
                                        "A": probs.get("away", 0.0)
                                    }
                                    rows.append(row)
                            proba_df = pd.DataFrame(rows, index=X_test.index[:len(rows)])
                        elif hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(self.X_test)
                            if isinstance(proba, dict) and '1x2' in proba:
                                proba = np.array(proba['1x2'])
                            proba_df = pd.DataFrame(
                                proba, index=X_test.index,
                                columns=["H", "D", "A"][:proba.shape[1] if hasattr(proba, 'shape') and len(proba.shape) > 1 else 3]
                            )
                        else:
                            logger.warning(f"  → Unknown prediction format from {name}: {type(preds)}")
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
        
        # Use the first model that can produce probabilities
        # Prefer Poisson for simulated bets as it's most stable
        model_name = next(iter(self.trained_models))
        model = self.trained_models[model_name]
        
        try:
            if not isinstance(self.test_df, pd.DataFrame):
                logger.warning(f"test_df is not a DataFrame: {type(self.test_df)}")
                return bets
                
            # Use pre-computed y_test (no re-fitting required)
            if self.y_test is None or self.X_test is None:
                logger.warning("Pre-computed test features not available")
                return bets
            
            y_test = self.y_test
            
            # Get probabilities — route to correct data source
            proba = None
            
            if model_name == "poisson":
                # Poisson needs raw DataFrame with team names
                preds = model.predict(self.test_df)
                if isinstance(preds, list) and len(preds) > 0 and hasattr(preds[0], 'probabilities'):
                    rows = []
                    for p in preds:
                        if p.market == Market.MATCH_WINNER:
                             probs = p.probabilities
                             rows.append([
                                 probs.get("home", 0.0),
                                 probs.get("draw", 0.0),
                                 probs.get("away", 0.0)
                             ])
                    proba = np.array(rows)
            else:
                # CatBoost / LightGBM use feature-engineered data
                preds = model.predict(self.X_test)
                if isinstance(preds, list) and len(preds) > 0 and hasattr(preds[0], 'probabilities'):
                    rows = []
                    for p in preds:
                        if p.market == Market.MATCH_WINNER:
                            probs = p.probabilities
                            rows.append([
                                probs.get("home", 0.0),
                                probs.get("draw", 0.0),
                                probs.get("away", 0.0)
                            ])
                    proba = np.array(rows)
                elif hasattr(model, 'predict_proba'):
                    p = model.predict_proba(self.X_test)
                    if isinstance(p, dict) and '1x2' in p:
                        proba = np.array(p['1x2'])
                    elif isinstance(p, np.ndarray):
                        proba = p
            
            if proba is None:
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
    
    def _run_walk_forward(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward validation with rolling windows."""
        logger.info("=" * 60)
        logger.info("Running Walk-Forward Validation")
        logger.info(f"Train: {self.config.wf_train_months}m, Test: {self.config.wf_test_months}m, Step: {self.config.wf_step_months}m")
        logger.info("=" * 60)
        
        # Ensure date sorting
        if "Date" in df.columns:
            df = df.sort_values("Date")
        
        # Parse Dates if needed
        if df["Date"].dtype == object:
             df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
            
        start_date = df["Date"].min()
        end_date = df["Date"].max()
        
        results = []
        fold = 0
        current_start = start_date
        
        # Metrics aggregation
        fold_metrics = {}
        
        while True:
            # Define window
            train_end = current_start + pd.DateOffset(months=self.config.wf_train_months)
            test_end = train_end + pd.DateOffset(months=self.config.wf_test_months)
            
            if test_end > end_date:
                break
            
            logger.info(f"\nFold {fold+1}: Train {current_start.date()} -> {train_end.date()}, Test -> {test_end.date()}")
            
            # Split data
            train_mask = (df["Date"] >= current_start) & (df["Date"] < train_end)
            test_mask = (df["Date"] >= train_end) & (df["Date"] < test_end)
            
            fold_train = df[train_mask].copy()
            fold_test = df[test_mask].copy()
            
            if len(fold_train) < 50 or len(fold_test) < 10:
                logger.warning(f"  Skipping fold (insufficient data: train={len(fold_train)}, test={len(fold_test)})")
                current_start += pd.DateOffset(months=self.config.wf_step_months)
                continue
                
            # Set context for training methods
            val_split_idx = int(len(fold_train) * 0.9)
            self.train_df = fold_train.iloc[:val_split_idx].copy()
            self.val_df = fold_train.iloc[val_split_idx:].copy()
            self.test_df = fold_test
            
            logger.info("  Building features...")
            X_train, y_train = self._build_features(self.train_df, fit_registry=True)
            X_val, y_val = self._build_features(self.val_df, fit_registry=False)
            X_test, y_test = self._build_features(self.test_df, fit_registry=False)
            
            logger.info(f"  Training models on {len(X_train)} samples...")
            model_results = self._train_models(X_train, y_train, X_val, y_val, X_test, y_test)
            
            # Aggregate fold results
            for name, res in model_results.items():
                if name not in fold_metrics:
                    fold_metrics[name] = {"roi": [], "accuracy": []}
                fold_metrics[name]["roi"].append(res.roi_simulated)
                fold_metrics[name]["accuracy"].append(res.accuracy)
                
                res_dict = res.to_dict()
                res_dict["fold"] = fold + 1
                res_dict["train_start"] = str(current_start.date())
                res_dict["test_end"] = str(test_end.date())
                results.append(res_dict)
            
            current_start += pd.DateOffset(months=self.config.wf_step_months)
            fold += 1
            
        logger.info("=" * 60)
        logger.info(f"Walk-Forward Complete: {fold} folds processed")
        
        summary = {}
        for name, metrics in fold_metrics.items():
            avg_roi = np.mean(metrics["roi"])
            std_roi = np.std(metrics["roi"])
            avg_acc = np.mean(metrics["accuracy"])
            logger.info(f"{name}: Avg ROI={avg_roi:+.2%} (±{std_roi:.2%}), Avg Acc={avg_acc:.2%}")
            summary[name] = {
                "avg_roi": avg_roi,
                "std_roi": std_roi, 
                "avg_accuracy": avg_acc
            }
            
        return {
            "walk_forward_results": results,
            "summary": summary
        }

    def _save_artifacts(self):
        """Save all training artifacts."""
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        
        # Save trained models
        for name, model in self.trained_models.items():
            try:
                model_path = output_dir / f"{name}.pkl"
                model.save(model_path)
                logger.info(f"  → Saved {name} model to {model_path}")
            except Exception as e:
                logger.error(f"Failed to save {name} model: {e}")

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
    walk_forward: bool = False,
) -> Dict[str, Any]:
    """Convenience function to run training pipeline."""
    # TODO: Pass walk_forward param to config when implemented
    if walk_forward:
        logger.info("Walk-forward validation enabled")
        
    config = TrainingConfig(
        data_path=Path(data_path),
        models=models or ["poisson", "catboost", "lightgbm"],
        walk_forward=walk_forward,
    )
    pipeline = TrainingPipeline(config=config)
    return pipeline.run()

if __name__ == "__main__":
    import argparse
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/historical.csv")
    parser.add_argument("--walk-forward", action="store_true", help="Enable walk-forward validation")
    args = parser.parse_args()
    
    # Use all models by default
    run_training_pipeline(
        data_path=args.data_path,
        models=["poisson", "catboost", "lightgbm"],
        walk_forward=args.walk_forward
    )
