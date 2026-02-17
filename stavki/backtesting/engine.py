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

from ..strategy.ev import EVResult

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
    
    Simulates betting on historical data using KellyStaker for realistic
    bankroll management and risk control.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        
        # Initialize Staker
        from ..strategy.kelly import KellyStaker
        
        staker_config = {
            "kelly_fraction": self.config.kelly_fraction,
            "max_stake_pct": self.config.max_stake_pct,
            # Map other config params if needed
        }
        self.staker = KellyStaker(bankroll=1000.0, config=staker_config)
    
    def run(
        self,
        data: pd.DataFrame,
        model_probs: Optional[Dict[str, np.ndarray]] = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data with vectorized signal generation.
        """
        logger.info(f"Running backtest on {len(data)} matches")
        
        # Reset staker for new run
        self.staker.bankroll = 1000.0
        self.staker.peak_bankroll = 1000.0
        self.staker.bet_history = []
        self.staker.pending_bets = []
        self.staker.daily_exposure.clear()
        self.staker.league_exposure.clear()
        
        # Ensure chronological order
        df = data.copy()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date")
        
        # Filter by leagues if specified
        if self.config.leagues and "League" in df.columns:
            df = df[df["League"].isin(self.config.leagues)]
            
        # 1. Vectorized Signal Generation
        # This replaces the row-by-row iteration for EV calculation
        signals = self._generate_signals(df, model_probs)
        
        logger.info(f"Generated {len(signals)} betting signals from {len(df)} matches")
        
        # 2. Sequential Execution (Staking)
        # We only iterate over potential bets, not all matches
        for row in signals.itertuples():
            # Extract date
            date = row.Date
            
            # Create EVResult from pre-calculated data
            ev_result = EVResult(
                match_id=row.match_id,
                market="match_winner",
                selection=row.selection,
                model_prob=row.model_prob,
                odds=row.odds,
                ev=row.ev,
                edge_pct=row.edge,
                implied_prob=row.implied_prob,
                bookmaker="Avg",
            )
            
            # Calculate stake (stateful)
            stake_result = self.staker.calculate_stake(
                ev_result=ev_result,
                league=row.League,
                date=date,
            )
            
            if stake_result.stake_amount > 0:
                # Place bet
                self.staker.place_bet(stake_result, row.League, date=date)
                
                # Apply slippage/bans
                if self.config.market_ban_prob > 0 and np.random.random() < self.config.market_ban_prob:
                    self.staker.settle_bet(ev_result.match_id, "void", date=date)
                    continue
                
                # Settle bet
                self.staker.settle_bet(ev_result.match_id, row.outcome, date=date)
        
        return self._compile_results()

    def _generate_signals(
        self,
        df: pd.DataFrame,
        model_probs: Optional[Dict[str, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Vectorized generation of betting signals.
        Returns DataFrame of OPPORTUNITIES (EV > min_ev).
        """
        # 1. Extract Odds
        odds_h = df.get("AvgOddsH", df.get("WHH", df.get("B365H", 0))).astype(float)
        odds_d = df.get("AvgOddsD", df.get("WHD", df.get("B365D", 0))).astype(float)
        odds_a = df.get("AvgOddsA", df.get("WHA", df.get("B365A", 0))).astype(float)
        
        # Valid odds mask
        valid_mask = (odds_h > 1) & (odds_d > 1) & (odds_a > 1)
        
        # 2. Align Model Probs
        n = len(df)
        if model_probs:
            # Convert dict to array aligned with df index
            # This handles the case where model_probs might be sparse or unordered
            # Assuming model_probs keys match df.index
            
            # Pre-allocate
            p_model = np.zeros((n, 3))
            
            # Identify indices present in both
            common_indices = df.index.intersection(list(model_probs.keys()))
            
            if len(common_indices) > 0:
                # This loop is technically iterating, but only for mapping
                # For huge datasets, we'd want model_probs to be a DataFrame already
                # But typically this dictionary lookup is fast enough compared to full logic
                # Optimization: if model_probs keys are exactly df.index, we can stack
                
                # Fast path: check if keys match exactly
                if len(model_probs) == n and np.array_equal(df.index, list(model_probs.keys())):
                     p_model = np.stack(list(model_probs.values()))
                else:
                    # Slow path: map
                    # Use reindexing if model_probs can be converted to DF
                    # Or simple loop for now (still faster than full logic)
                    # For safety in this refactor, let's use a safe mapping
                    # Create a Series of arrays? No, 2D array is better for math.
                    
                    # Create temporary DF to align
                    probs_df = pd.DataFrame.from_dict(model_probs, orient='index')
                    probs_df = probs_df.reindex(df.index).fillna(0)
                    p_model = probs_df.values
            
        else:
            # Market implied baseline
            inv_h, inv_d, inv_a = 1/odds_h, 1/odds_d, 1/odds_a
            total_inv = inv_h + inv_d + inv_a
            p_model = np.column_stack([inv_h/total_inv, inv_d/total_inv, inv_a/total_inv])
            # Handle zeros (inf)
            p_model = np.nan_to_num(p_model, 0.0)

        # 3. Market Implied Probs (Normalized)
        inv_h, inv_d, inv_a = 1/odds_h, 1/odds_d, 1/odds_a
        total_inv = inv_h + inv_d + inv_a
        p_market = np.column_stack([inv_h/total_inv, inv_d/total_inv, inv_a/total_inv])
        
        # 4. Blend
        alpha = self.config.model_alpha
        p_blended = alpha * p_model + (1 - alpha) * p_market
        
        # 5. Calculate EVs
        # odds_arr shape: (N, 3)
        odds_arr = np.column_stack([odds_h, odds_d, odds_a])
        evs = p_blended * odds_arr - 1
        
        # 6. Find Best Bet per Row
        best_idx = np.argmax(evs, axis=1)
        # Advanced indexing to get values
        rows = np.arange(n)
        best_ev = evs[rows, best_idx]
        best_prob = p_blended[rows, best_idx]
        best_implied = p_market[rows, best_idx]
        best_odds = odds_arr[rows, best_idx]
        
        edge = best_prob - best_implied
        
        # 7. Apply Filters
        mask = (valid_mask) & \
               (best_ev >= self.config.min_ev) & \
               (edge >= self.config.min_edge)
               
        if not mask.any():
            return pd.DataFrame()
            
        # 8. Construct Result DataFrame
        # Get outcomes
        ftr = df.get("FTR", df.get("Result", "")).map({"H": 0, "D": 1, "A": 2}).fillna(-1).astype(int)
        actual_outcomes = np.where(ftr == best_idx, "win", "loss")
        
        # Selection strings
        selections = np.array(["home", "draw", "away"])
        best_selections = selections[best_idx]
        
        # Build signals DF
        signals = df[mask].copy()
        signals["selection"] = best_selections[mask]
        signals["model_prob"] = best_prob[mask]
        signals["odds"] = best_odds[mask]
        signals["ev"] = best_ev[mask]
        signals["edge"] = edge[mask]
        signals["outcome"] = actual_outcomes[mask]
        signals["implied_prob"] = best_implied[mask]
        
        # Ensure match_id exists
        if "match_id" not in signals.columns:
            from stavki.utils import generate_match_id
            # Vectorized ID generation is tricky with the util fn
            # But we can iterate just for ID if needed, or use apply
            # Since we filtered down significantly, apply is fine
            signals["match_id"] = signals.apply(
                lambda x: generate_match_id(x.get("HomeTeam", ""), x.get("AwayTeam", ""), x.get("Date")),
                axis=1
            )
            
        return signals

    def _compile_results(self) -> BacktestResult:
        """Helper to compile final stats from staker."""
        stats = self.staker.get_stats()
        
        # Calculate Sharpe from history
        returns = []
        bets = self.staker.bet_history
        for b in bets:
            if b.get("stake", 0) > 0:
                returns.append(b.get("profit", 0) / b["stake"])
        
        if len(returns) > 1:
            avg_ret = np.mean(returns)
            std_ret = np.std(returns)
            sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
        else:
            sharpe = 0
            
        # Reconstruct league stats
        league_stats = {}
        for b in bets:
            l = b.get("league", "Unknown")
            if l not in league_stats:
                league_stats[l] = {"bets": 0, "wins": 0, "profit": 0.0, "stake": 0.0}
            
            league_stats[l]["bets"] += 1
            league_stats[l]["stake"] += b.get("stake", 0)
            league_stats[l]["profit"] += b.get("profit", 0)
            if b.get("status") == "win":
                league_stats[l]["wins"] += 1

        for l in league_stats:
            s = league_stats[l]
            s["roi"] = s["profit"] / s["stake"] if s["stake"] > 0 else 0
            s["win_rate"] = s["wins"] / s["bets"] if s["bets"] > 0 else 0
            
        return BacktestResult(
            total_bets=stats["total_bets"],
            winning_bets=stats["wins"],
            total_stake=stats["total_staked"],
            total_profit=stats["total_profit"],
            roi=stats["roi"],
            win_rate=stats["win_rate"],
            max_drawdown=stats["drawdown"],
            sharpe_ratio=sharpe,
            avg_odds=np.mean([b["odds"] for b in bets]) if bets else 0,
            avg_ev=np.mean([b["ev"] for b in bets]) if bets else 0,
            league_results=league_stats,
            bet_history=bets,
        )

    # Legacy method kept but unused in vectorized run
    def _evaluate_match(self, *args, **kwargs):
        pass


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
