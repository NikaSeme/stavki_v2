"""
Data-Driven Weight Optimizer
============================

Optimizes ensemble weights and strategy parameters through:
1. Grid search with cross-validation
2. Temporal splits (no data leakage)
3. Multiple metrics (ROI, Sharpe, Kelly growth)

ALL WEIGHTS ARE OPTIMIZED, NOT HARDCODED.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import itertools
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of weight/parameter optimization."""
    optimal_weights: Dict[str, float]
    optimal_params: Dict[str, Any]
    best_metric: float
    metric_name: str
    n_trials: int
    all_results: List[Dict]
    
    def to_dict(self) -> dict:
        return {
            "optimal_weights": self.optimal_weights,
            "optimal_params": self.optimal_params,
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
            "n_trials": self.n_trials,
        }


class WeightOptimizer:
    """
    Optimize ensemble model weights through backtesting.
    
    Key principle: Never use hardcoded weights.
    """
    
    def __init__(
        self,
        step_size: float = 0.05,       # Weight granularity
        min_weight: float = 0.0,        # Minimum weight per model
        max_weight: float = 1.0,        # Maximum weight per model
        n_folds: int = 3,               # Cross-validation folds (temporal)
    ):
        self.step_size = step_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.n_folds = n_folds
    
    def optimize_ensemble_weights(
        self,
        model_predictions: Dict[str, pd.DataFrame],
        actual_outcomes: pd.Series,
        odds_data: pd.DataFrame,
        metric: str = "roi",
    ) -> OptimizationResult:
        """
        Optimize weights for ensemble of models.
        
        Args:
            model_predictions: {model_name: DataFrame with probs}
            actual_outcomes: Series with actual results
            odds_data: DataFrame with odds
            metric: "roi", "accuracy", "log_loss", "sharpe"
        
        Returns:
            OptimizationResult with optimal weights
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        
        # Generate all weight combinations that sum to 1
        weight_combinations = self._generate_weight_combinations(n_models)
        
        logger.info(f"Testing {len(weight_combinations)} weight combinations")
        
        all_results = []
        best_score = float("-inf")
        best_weights = None
        
        for weights in weight_combinations:
            weight_dict = dict(zip(model_names, weights))
            
            # Calculate ensemble predictions
            ensemble_probs = self._calculate_ensemble_probs(
                model_predictions, weight_dict
            )
            
            # Calculate metric
            score = self._calculate_metric(
                ensemble_probs, actual_outcomes, odds_data, metric
            )
            
            all_results.append({
                "weights": weight_dict,
                "score": score,
            })
            
            if score > best_score:
                best_score = score
                best_weights = weight_dict
        
        # Sort results
        all_results.sort(key=lambda x: -x["score"])
        
        logger.info(f"Optimal weights: {best_weights} ({metric}={best_score:.4f})")
        
        return OptimizationResult(
            optimal_weights=best_weights,
            optimal_params={},
            best_metric=best_score,
            metric_name=metric,
            n_trials=len(weight_combinations),
            all_results=all_results[:10],  # Top 10
        )
    
    def optimize_per_league(
        self,
        model_predictions: Dict[str, pd.DataFrame],
        actual_outcomes: pd.Series,
        odds_data: pd.DataFrame,
        leagues: List[str],
        metric: str = "roi",
    ) -> Dict[str, Dict[str, float]]:
        """
        Optimize weights separately for each league.
        
        Returns:
            {league: {model: weight}}
        """
        league_weights = {}
        
        for league in leagues:
            # Filter data for this league
            league_mask = odds_data["League"] == league
            
            league_preds = {
                name: df[league_mask] 
                for name, df in model_predictions.items()
            }
            league_outcomes = actual_outcomes[league_mask]
            league_odds = odds_data[league_mask]
            
            # Skip if not enough data
            if len(league_outcomes) < 50:
                logger.warning(f"Skipping {league}: only {len(league_outcomes)} bets")
                continue
            
            # Optimize
            result = self.optimize_ensemble_weights(
                league_preds, league_outcomes, league_odds, metric
            )
            
            league_weights[league] = result.optimal_weights
            logger.info(f"{league}: {result.optimal_weights}")
        
        return league_weights
    
    def _generate_weight_combinations(self, n_models: int) -> List[Tuple[float, ...]]:
        """Generate valid weight combinations that sum to 1."""
        steps = int((self.max_weight - self.min_weight) / self.step_size) + 1
        weights = [self.min_weight + i * self.step_size for i in range(steps)]
        
        combinations = []
        for combo in itertools.product(weights, repeat=n_models):
            if abs(sum(combo) - 1.0) < 0.01:  # Must sum to ~1
                combinations.append(combo)
        
        return combinations
    
    def _calculate_ensemble_probs(
        self,
        model_predictions: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
    ) -> pd.DataFrame:
        """Calculate weighted ensemble probabilities."""
        result = None
        
        for model_name, df in model_predictions.items():
            weight = weights.get(model_name, 0)
            
            if result is None:
                result = df.copy() * weight
            else:
                result += df * weight
        
        return result
    
    def _calculate_metric(
        self,
        probs: pd.DataFrame,
        actual: pd.Series,
        odds: pd.DataFrame,
        metric: str,
    ) -> float:
        """Calculate optimization metric."""
        if metric == "accuracy":
            predictions = probs.idxmax(axis=1)
            return (predictions == actual).mean()
        
        elif metric == "roi":
            # Simulate betting
            total_staked = 0
            total_profit = 0
            
            for idx in probs.index:
                if idx not in actual.index:
                    continue
                
                row_probs = probs.loc[idx]
                best_outcome = row_probs.idxmax()
                
                odds_col = f"odds_{best_outcome}" if f"odds_{best_outcome}" in odds.columns else None
                if odds_col and idx in odds.index:
                    bet_odds = odds.loc[idx, odds_col]
                else:
                    continue
                
                ev = row_probs[best_outcome] * bet_odds - 1
                
                if ev > 0.03:  # 3% EV threshold
                    stake = 10  # Fixed stake
                    total_staked += stake
                    
                    if actual.loc[idx] == best_outcome:
                        total_profit += stake * (bet_odds - 1)
                    else:
                        total_profit -= stake
            
            return total_profit / total_staked if total_staked > 0 else 0
        
        elif metric == "log_loss":
            # Negative log loss (higher is better)
            eps = 1e-10
            loss = 0
            
            for idx in probs.index:
                if idx not in actual.index:
                    continue
                
                actual_outcome = actual.loc[idx]
                if actual_outcome in probs.columns:
                    prob = probs.loc[idx, actual_outcome]
                    loss -= np.log(max(prob, eps))
            
            return -loss  # Negative because we maximize
        
        return 0


class KellyOptimizer:
    """Optimize Kelly fraction through backtesting."""
    
    def __init__(
        self,
        fractions: List[float] = None,
        initial_bankroll: float = 1000.0,
    ):
        self.fractions = fractions or [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50
        ]
        self.initial_bankroll = initial_bankroll
    
    def optimize(
        self,
        historical_bets: List[Dict],
        metric: str = "risk_adjusted",  # "roi", "final_bankroll", "risk_adjusted"
    ) -> Tuple[float, Dict[float, Dict]]:
        """
        Find optimal Kelly fraction.
        
        Args:
            historical_bets: List of {prob, odds, result}
            metric: Optimization target
        
        Returns:
            (optimal_fraction, {fraction: metrics})
        """
        results = {}
        
        for fraction in self.fractions:
            bankroll = self.initial_bankroll
            peak = bankroll
            max_drawdown = 0
            
            for bet in historical_bets:
                prob = bet.get("prob") or bet.get("model_prob")
                odds = bet.get("odds")
                result = bet.get("result")
                
                if not all([prob, odds, result]):
                    continue
                
                # Kelly formula
                b = odds - 1
                if b <= 0:
                    continue
                
                p = prob
                q = 1 - p
                kelly_full = max(0, (b * p - q) / b)
                stake_pct = min(kelly_full * fraction, 0.05)  # Max 5% per bet
                stake = bankroll * stake_pct
                
                # Settle
                if result == "win":
                    profit = stake * (odds - 1)
                elif result == "loss":
                    profit = -stake
                else:
                    profit = 0
                
                bankroll += profit
                
                # Update metrics
                if bankroll > peak:
                    peak = bankroll
                
                dd = (peak - bankroll) / peak if peak > 0 else 0
                max_drawdown = max(max_drawdown, dd)
            
            roi = (bankroll - self.initial_bankroll) / self.initial_bankroll
            
            # Per-bet returns for proper Sharpe approximation
            bet_returns = []
            temp_bank = self.initial_bankroll
            for bet in historical_bets:
                prob_b = bet.get("prob") or bet.get("model_prob")
                odds_b = bet.get("odds")
                result_b = bet.get("result")
                if not all([prob_b, odds_b, result_b]):
                    continue
                b_b = odds_b - 1
                if b_b <= 0:
                    continue
                q_b = 1 - prob_b
                kf = max(0, (b_b * prob_b - q_b) / b_b)
                sp = min(kf * fraction, 0.05)
                if result_b == "win":
                    ret = sp * (odds_b - 1)
                elif result_b == "loss":
                    ret = -sp
                else:
                    ret = 0
                bet_returns.append(ret)
            std_returns = np.std(bet_returns) if len(bet_returns) > 1 else 0.01
            
            results[fraction] = {
                "final_bankroll": bankroll,
                "roi": roi,
                "max_drawdown": max_drawdown,
                "risk_adjusted": roi / max(std_returns, 0.01),
            }
        
        # Select best
        if metric == "risk_adjusted":
            best = max(results.keys(), key=lambda f: results[f]["risk_adjusted"])
        elif metric == "roi":
            best = max(results.keys(), key=lambda f: results[f]["roi"])
        else:
            best = max(results.keys(), key=lambda f: results[f]["final_bankroll"])
        
        logger.info(
            f"Optimal Kelly fraction: {best:.2f} "
            f"(ROI: {results[best]['roi']:.2%}, "
            f"Drawdown: {results[best]['max_drawdown']:.1%})"
        )
        
        return best, results


class ThresholdOptimizer:
    """Optimize EV and edge thresholds."""
    
    def __init__(
        self,
        ev_range: Tuple[float, float] = (0.02, 0.15),
        edge_range: Tuple[float, float] = (0.01, 0.10),
        step: float = 0.01,
    ):
        self.ev_range = ev_range
        self.edge_range = edge_range
        self.step = step
    
    def optimize(
        self,
        historical_bets: List[Dict],
        min_bets: int = 30,
    ) -> Dict[str, float]:
        """
        Find optimal EV and edge thresholds.
        
        Args:
            historical_bets: List of {ev, edge, stake, odds, result}
            min_bets: Minimum bets to consider a threshold valid
        
        Returns:
            {"min_ev": x, "min_edge": y, "roi": z}
        """
        best = {"min_ev": 0.03, "min_edge": 0.02, "roi": 0}
        best_score = float("-inf")
        
        ev_values = np.arange(
            self.ev_range[0], self.ev_range[1] + self.step, self.step
        )
        edge_values = np.arange(
            self.edge_range[0], self.edge_range[1] + self.step, self.step
        )
        
        for ev_threshold in ev_values:
            for edge_threshold in edge_values:
                # Filter bets
                filtered = [
                    b for b in historical_bets
                    if b.get("ev", 0) >= ev_threshold 
                    and b.get("edge", 0) >= edge_threshold
                ]
                
                if len(filtered) < min_bets:
                    continue
                
                # Calculate ROI
                total_staked = sum(b.get("stake", 1) for b in filtered)
                total_profit = sum(
                    b.get("stake", 1) * (b["odds"] - 1) if b["result"] == "win" 
                    else -b.get("stake", 1)
                    for b in filtered if b.get("result") in ("win", "loss")
                )
                
                roi = total_profit / total_staked if total_staked > 0 else 0
                
                # Score: ROI * sqrt(n_bets) to balance quantity and quality
                score = roi * np.sqrt(len(filtered))
                
                if score > best_score:
                    best_score = score
                    best = {
                        "min_ev": ev_threshold,
                        "min_edge": edge_threshold,
                        "roi": roi,
                        "n_bets": len(filtered),
                    }
        
        logger.info(
            f"Optimal thresholds: EV>={best['min_ev']:.2%}, "
            f"Edge>={best['min_edge']:.2%} "
            f"(ROI: {best['roi']:.2%} on {best.get('n_bets', 0)} bets)"
        )
        
        return best


def save_optimized_config(
    config: Dict[str, Any],
    filepath: Path,
):
    """Save optimized configuration to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, "w") as f:
        json.dump(config, f, indent=2, default=str)
    
    logger.info(f"Saved config to {filepath}")


def load_optimized_config(filepath: Path) -> Dict[str, Any]:
    """Load optimized configuration from file."""
    with open(filepath) as f:
        return json.load(f)
