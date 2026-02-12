"""
Backtesting Engine
==================

Core backtesting infrastructure for STAVKI:
1. BacktestEngine - Main simulation engine
2. WalkForwardValidator - Rolling temporal cross-validation
3. MonteCarloSimulator - Statistical confidence analysis
4. RealitySimulator - Adjusts for real-world conditions

Usage:
    engine = BacktestEngine(config)
    result = engine.run(historical_data)
    print(f"ROI: {result.roi:.2%}")
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Strategy parameters
    min_ev: float = 0.05
    min_edge: float = 0.02
    kelly_fraction: float = 0.25
    max_stake_pct: float = 0.05
    
    # Data parameters
    leagues: List[str] = field(default_factory=lambda: ["EPL"])
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Model blending
    model_alpha: float = 0.50  # Weight given to model (vs market)
    
    # Monte Carlo
    n_simulations: int = 10000
    confidence_level: float = 0.95
    
    # Walk-Forward
    train_months: int = 6
    test_months: int = 2
    step_months: int = 1
    
    # Reality adjustments
    slippage: float = 0.02  # 2% odds reduction
    market_ban_prob: float = 0.05  # 5% chance of stake rejection


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    total_bets: int
    winning_bets: int
    total_stake: float
    total_profit: float
    roi: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    
    # Optional extended metrics
    roi_ci_lower: float = 0.0
    roi_ci_upper: float = 0.0
    avg_odds: float = 0.0
    avg_ev: float = 0.0
    
    # Per-league breakdown
    league_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Bet history
    bet_history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "total_bets": self.total_bets,
            "winning_bets": self.winning_bets,
            "total_stake": round(self.total_stake, 2),
            "total_profit": round(self.total_profit, 2),
            "roi": round(self.roi, 4),
            "win_rate": round(self.win_rate, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "roi_ci_lower": round(self.roi_ci_lower, 4),
            "roi_ci_upper": round(self.roi_ci_upper, 4),
            "avg_odds": round(self.avg_odds, 3),
            "avg_ev": round(self.avg_ev, 4),
            "league_results": self.league_results,
        }


class BacktestEngine:
    """
    Main backtesting engine.
    
    Simulates betting on historical data, tracking PnL, drawdown, etc.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
    
    def run(
        self,
        data: pd.DataFrame,
        model_probs: Optional[Dict[str, np.ndarray]] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            data: DataFrame with historical matches and odds.
                  Required columns: HomeTeam, AwayTeam, FTR, AvgOddsH, AvgOddsD, AvgOddsA
            model_probs: Pre-computed model probabilities (optional)
        
        Returns:
            BacktestResult with all metrics
        """
        logger.info(f"Running backtest on {len(data)} matches")
        
        # Initialize tracking
        bankroll = 1000.0
        initial_bankroll = bankroll
        peak_bankroll = bankroll
        max_drawdown = 0.0
        
        bets = []
        league_stats = {}
        
        returns = []  # For Sharpe calculation
        
        # Filter by leagues if specified
        if self.config.leagues and "League" in data.columns:
            data = data[data["League"].isin(self.config.leagues)]
        
        # Process each match
        for idx, row in data.iterrows():
            result = self._evaluate_match(row, model_probs, idx)
            
            if result is None:
                continue
            
            bet_outcome, stake, odds, ev, league = result
            
            # Calculate profit
            if bet_outcome == "win":
                profit = stake * (odds - 1)
            elif bet_outcome == "loss":
                profit = -stake
            else:
                profit = 0
            
            # Apply slippage
            if self.config.slippage > 0:
                profit *= (1 - self.config.slippage)
            
            # Update bankroll
            bankroll += profit
            
            # Track drawdown
            if bankroll > peak_bankroll:
                peak_bankroll = bankroll
            
            current_dd = (peak_bankroll - bankroll) / peak_bankroll
            max_drawdown = max(max_drawdown, current_dd)
            
            # Track returns
            returns.append(profit / stake if stake > 0 else 0)
            
            # Record bet
            bet_record = {
                "match": f"{row.get('HomeTeam', '')} vs {row.get('AwayTeam', '')}",
                "league": league,
                "odds": odds,
                "stake": stake,
                "ev": ev,
                "outcome": bet_outcome,
                "profit": profit,
                "bankroll_after": bankroll,
            }
            bets.append(bet_record)
            
            # Track per-league
            if league not in league_stats:
                league_stats[league] = {"bets": 0, "wins": 0, "profit": 0, "stake": 0}
            
            league_stats[league]["bets"] += 1
            league_stats[league]["stake"] += stake
            league_stats[league]["profit"] += profit
            if bet_outcome == "win":
                league_stats[league]["wins"] += 1
        
        # Calculate final metrics
        total_bets = len(bets)
        winning_bets = sum(1 for b in bets if b["outcome"] == "win")
        total_stake = sum(b["stake"] for b in bets)
        total_profit = bankroll - initial_bankroll
        
        roi = total_profit / total_stake if total_stake > 0 else 0
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        # Sharpe ratio
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # Per-league results
        for league in league_stats:
            ls = league_stats[league]
            ls["roi"] = ls["profit"] / ls["stake"] if ls["stake"] > 0 else 0
            ls["win_rate"] = ls["wins"] / ls["bets"] if ls["bets"] > 0 else 0
        
        result = BacktestResult(
            total_bets=total_bets,
            winning_bets=winning_bets,
            total_stake=total_stake,
            total_profit=total_profit,
            roi=roi,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            avg_odds=np.mean([b["odds"] for b in bets]) if bets else 0,
            avg_ev=np.mean([b["ev"] for b in bets]) if bets else 0,
            league_results=league_stats,
            bet_history=bets,
        )
        
        logger.info(f"Backtest complete: {total_bets} bets, ROI={roi:.2%}")
        
        return result
    
    def _evaluate_match(
        self,
        row: pd.Series,
        model_probs: Optional[Dict],
        idx: Any,
    ) -> Optional[Tuple[str, float, float, float, str]]:
        """
        Evaluate a single match for betting.
        
        Returns:
            (outcome, stake, odds, ev, league) or None if no bet
        """
        # Get odds
        odds_h = row.get("AvgOddsH", row.get("WHH", row.get("B365H", 0)))
        odds_d = row.get("AvgOddsD", row.get("WHD", row.get("B365D", 0)))
        odds_a = row.get("AvgOddsA", row.get("WHA", row.get("B365A", 0)))
        
        if not all([odds_h, odds_d, odds_a]) or any(o <= 1 for o in [odds_h, odds_d, odds_a]):
            return None
        
        # Get model probabilities
        if model_probs is not None and idx in model_probs:
            p_model = model_probs[idx]
        else:
            # Use market-implied probabilities as baseline
            implied = [1/odds_h, 1/odds_d, 1/odds_a]
            total = sum(implied)
            p_model = np.array([p/total for p in implied])
        
        # Blend with market
        implied_probs = np.array([1/odds_h, 1/odds_d, 1/odds_a])
        implied_probs /= implied_probs.sum()
        
        alpha = self.config.model_alpha
        blended = alpha * p_model + (1 - alpha) * implied_probs
        
        # Calculate EVs
        odds_arr = np.array([odds_h, odds_d, odds_a])
        evs = blended * odds_arr - 1
        
        # Find best bet
        best_idx = np.argmax(evs)
        best_ev = evs[best_idx]
        
        if best_ev < self.config.min_ev:
            return None
        
        # Check edge
        edge = blended[best_idx] - implied_probs[best_idx]
        if edge < self.config.min_edge:
            return None
        
        # Calculate stake (Kelly)
        kelly_full = (blended[best_idx] * odds_arr[best_idx] - 1) / (odds_arr[best_idx] - 1)
        stake_pct = min(
            kelly_full * self.config.kelly_fraction,
            self.config.max_stake_pct
        )
        stake = 1000 * stake_pct  # Assume 1000 bankroll
        
        if stake < 1:  # Minimum stake
            return None
        
        # Determine actual outcome
        actual_result = row.get("FTR", row.get("Result", ""))
        result_map = {"H": 0, "D": 1, "A": 2}
        actual_idx = result_map.get(actual_result, -1)
        
        if actual_idx == -1:
            return None
        
        outcome = "win" if actual_idx == best_idx else "loss"
        
        league = row.get("League", row.get("Div", "Unknown"))
        
        return (outcome, stake, odds_arr[best_idx], best_ev, league)


class WalkForwardValidator:
    """
    Walk-Forward validation for temporal cross-validation.
    
    Trains on N months, tests on M months, slides by S months.
    """
    
    def __init__(
        self,
        train_months: int = 6,
        test_months: int = 2,
        step_months: int = 1,
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
    
    def validate(
        self,
        data: pd.DataFrame,
        engine: BacktestEngine,
        date_col: str = "Date",
    ) -> List[BacktestResult]:
        """
        Run walk-forward validation.
        
        Args:
            data: Historical data with date column
            engine: BacktestEngine instance
            date_col: Name of date column
        
        Returns:
            List of BacktestResult for each fold
        """
        # Ensure date column is datetime
        if date_col in data.columns:
            data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
            data = data.dropna(subset=[date_col])
            data = data.sort_values(date_col)
        else:
            logger.warning(f"No {date_col} column, using index-based splits")
            return self._validate_by_index(data, engine)
        
        results = []
        
        min_date = data[date_col].min()
        max_date = data[date_col].max()
        
        current_start = min_date
        fold = 0
        
        while True:
            # Training period
            train_end = current_start + pd.DateOffset(months=self.train_months)
            test_end = train_end + pd.DateOffset(months=self.test_months)
            
            if test_end > max_date:
                break
            
            # Split data
            train_data = data[(data[date_col] >= current_start) & (data[date_col] < train_end)]
            test_data = data[(data[date_col] >= train_end) & (data[date_col] < test_end)]
            
            if len(test_data) < 10:
                current_start += pd.DateOffset(months=self.step_months)
                continue
            
            logger.info(f"Fold {fold}: train={len(train_data)}, test={len(test_data)}")
            
            # Run backtest on test period
            result = engine.run(test_data)
            results.append(result)
            
            # Move window
            current_start += pd.DateOffset(months=self.step_months)
            fold += 1
        
        logger.info(f"Walk-Forward: {len(results)} folds completed")
        
        return results
    
    def _validate_by_index(
        self,
        data: pd.DataFrame,
        engine: BacktestEngine,
    ) -> List[BacktestResult]:
        """Fallback: split by index."""
        n = len(data)
        fold_size = n // 5  # 5 folds
        
        results = []
        
        for i in range(5):
            start_idx = i * fold_size
            end_idx = (i + 1) * fold_size if i < 4 else n
            
            fold_data = data.iloc[start_idx:end_idx]
            
            if len(fold_data) < 10:
                continue
            
            result = engine.run(fold_data)
            results.append(result)
        
        return results
    
    def get_combined_result(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """Combine results from all folds."""
        if not results:
            return {}
        
        total_bets = sum(r.total_bets for r in results)
        total_stake = sum(r.total_stake for r in results)
        total_profit = sum(r.total_profit for r in results)
        
        rois = [r.roi for r in results if r.total_bets > 0]
        positive_folds = sum(1 for r in rois if r > 0)
        
        return {
            "n_folds": len(results),
            "aggregate_roi": total_profit / total_stake if total_stake > 0 else 0,
            "total_bets": total_bets,
            "total_profit": total_profit,
            "consistency": positive_folds / len(rois) if rois else 0,
            "avg_roi": np.mean(rois) if rois else 0,
            "std_roi": np.std(rois) if rois else 0,
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for confidence intervals and risk metrics.
    
    Randomly resamples bet history to estimate ROI distribution.
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
    ):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level
    
    def simulate(self, result: BacktestResult) -> Dict[str, float]:
        """
        Run Monte Carlo simulation on backtest results.
        
        Args:
            result: BacktestResult with bet history
        
        Returns:
            Dict with confidence intervals, VaR, etc.
        """
        if not result.bet_history:
            return {}
        
        # Extract returns (profit / stake for each bet)
        returns = []
        for bet in result.bet_history:
            stake = bet.get("stake", 1)
            profit = bet.get("profit", 0)
            if stake > 0:
                returns.append(profit / stake)
        
        if len(returns) < 2:
            return {}
        
        returns = np.array(returns)
        n_bets = len(returns)
        
        # Run simulations
        simulated_rois = []
        simulated_drawdowns = []
        
        for _ in range(self.n_simulations):
            # Bootstrap sample with replacement
            sampled = np.random.choice(returns, size=n_bets, replace=True)
            
            # Calculate ROI of this simulation
            roi = np.mean(sampled)
            simulated_rois.append(roi)
            
            # Calculate max drawdown
            cumsum = np.cumsum(sampled)
            peak = np.maximum.accumulate(cumsum)
            drawdown = (peak - cumsum) / (peak + 1e-10)  # Avoid div by zero
            simulated_drawdowns.append(np.max(drawdown))
        
        simulated_rois = np.array(simulated_rois)
        simulated_drawdowns = np.array(simulated_drawdowns)
        
        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(simulated_rois, alpha / 2 * 100)
        ci_upper = np.percentile(simulated_rois, (1 - alpha / 2) * 100)
        
        # Calculate VaR (Value at Risk)
        var_5 = np.percentile(simulated_rois, 5)
        var_1 = np.percentile(simulated_rois, 1)
        
        # Probability of positive ROI
        prob_positive = np.mean(simulated_rois > 0)
        
        return {
            "roi_ci_lower": ci_lower,
            "roi_ci_upper": ci_upper,
            "roi_mean": np.mean(simulated_rois),
            "roi_median": np.median(simulated_rois),
            "roi_std": np.std(simulated_rois),
            "prob_positive_roi": prob_positive,
            "var_5": var_5,
            "var_1": var_1,
            "max_drawdown_mean": np.mean(simulated_drawdowns),
            "max_drawdown_95": np.percentile(simulated_drawdowns, 95),
        }


class RealitySimulator:
    """
    Adjusts backtest parameters for real-world conditions.
    
    Scenarios:
    - optimistic: Best case (no slippage)
    - realistic: Normal conditions
    - pessimistic: Cautious estimates
    - worst_case: Stress test
    """
    
    SCENARIOS = {
        "optimistic": {
            "slippage": 0.00,
            "market_ban_prob": 0.02,
            "odds_reduction": 0.00,
        },
        "realistic": {
            "slippage": 0.02,
            "market_ban_prob": 0.05,
            "odds_reduction": 0.01,
        },
        "pessimistic": {
            "slippage": 0.05,
            "market_ban_prob": 0.10,
            "odds_reduction": 0.03,
        },
        "worst_case": {
            "slippage": 0.10,
            "market_ban_prob": 0.20,
            "odds_reduction": 0.05,
        },
    }
    
    def __init__(self, scenario: str = "realistic"):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        self.scenario = scenario
        self.params = self.SCENARIOS[scenario]
    
    def adjust_config(self, config: BacktestConfig) -> BacktestConfig:
        """Adjust config for reality scenario."""
        config.slippage = self.params["slippage"]
        config.market_ban_prob = self.params["market_ban_prob"]
        
        return config
    
    def adjust_odds(self, odds: float) -> float:
        """Adjust odds for realistic execution."""
        reduction = self.params["odds_reduction"]
        return odds * (1 - reduction)


def run_backtest(
    data_path: str,
    leagues: List[str] = None,
    min_ev: float = 0.05,
    kelly: float = 0.25,
) -> BacktestResult:
    """Convenience function to run a backtest."""
    data = pd.read_csv(data_path)
    
    config = BacktestConfig(
        leagues=leagues or [],
        min_ev=min_ev,
        kelly_fraction=kelly,
    )
    
    engine = BacktestEngine(config=config)
    return engine.run(data)
