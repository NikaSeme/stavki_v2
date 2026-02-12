"""
Backtesting Metrics Calculator
==============================

Calculates comprehensive metrics for backtest analysis:
- ROI, Sharpe, Sortino, Calmar
- Win rate, yield, CLV
- Per-outcome and per-league breakdowns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MetricsSummary:
    """Summary of all calculated metrics."""
    # Core metrics
    roi: float
    win_rate: float
    avg_odds: float
    avg_ev: float
    
    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    
    # Betting metrics
    yield_pct: float  # profit / number of bets
    clv: float  # Closing Line Value
    
    # Distribution
    median_return: float
    return_std: float
    skewness: float
    kurtosis: float
    
    def to_dict(self) -> dict:
        return {
            "roi": round(self.roi, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_odds": round(self.avg_odds, 3),
            "avg_ev": round(self.avg_ev, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "yield_pct": round(self.yield_pct, 4),
            "clv": round(self.clv, 4),
            "median_return": round(self.median_return, 4),
            "return_std": round(self.return_std, 4),
            "skewness": round(self.skewness, 3),
            "kurtosis": round(self.kurtosis, 3),
        }


class MetricsCalculator:
    """
    Calculate comprehensive betting metrics.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_all(
        self,
        bet_history: List[Dict],
        closing_odds: Optional[Dict[str, float]] = None,
    ) -> MetricsSummary:
        """
        Calculate all metrics from bet history.
        
        Args:
            bet_history: List of bet records with profit, stake, odds
            closing_odds: Optional closing odds for CLV calculation
        
        Returns:
            MetricsSummary with all metrics
        """
        if not bet_history:
            return self._empty_metrics()
        
        # Extract data
        profits = np.array([b.get("profit", 0) for b in bet_history])
        stakes = np.array([b.get("stake", 1) for b in bet_history])
        odds = np.array([b.get("odds", 1) for b in bet_history])
        evs = np.array([b.get("ev", 0) for b in bet_history])
        
        # Calculate returns (profit / stake)
        returns = np.where(stakes > 0, profits / stakes, 0)
        
        # Core metrics
        total_profit = profits.sum()
        total_stake = stakes.sum()
        roi = total_profit / total_stake if total_stake > 0 else 0
        
        wins = sum(1 for b in bet_history if b.get("outcome") == "win")
        win_rate = wins / len(bet_history)
        
        avg_odds = odds.mean()
        avg_ev = evs.mean()
        
        # Risk metrics
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)
        max_dd = self._calculate_max_drawdown(profits)
        calmar = roi / max_dd if max_dd > 0 else 0
        
        # Betting metrics
        yield_pct = total_profit / len(bet_history)
        
        # CLV (Closing Line Value)
        clv = self._calculate_clv(bet_history, closing_odds) if closing_odds else 0
        
        # Distribution
        median_return = np.median(returns)
        return_std = np.std(returns)
        skewness = self._calculate_skewness(returns)
        kurtosis = self._calculate_kurtosis(returns)
        
        return MetricsSummary(
            roi=roi,
            win_rate=win_rate,
            avg_odds=avg_odds,
            avg_ev=avg_ev,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            yield_pct=yield_pct,
            clv=clv,
            median_return=median_return,
            return_std=return_std,
            skewness=skewness,
            kurtosis=kurtosis,
        )
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Sharpe ratio: (return - risk_free) / std."""
        if len(returns) < 2:
            return 0
        
        avg_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0
        
        # Annualize (assume daily returns, 252 trading days)
        sharpe = (avg_return - self.risk_free_rate / 252) / std_return * np.sqrt(252)
        
        return sharpe
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Sortino ratio: uses downside deviation instead of std."""
        if len(returns) < 2:
            return 0
        
        avg_return = returns.mean()
        
        # Downside deviation (only negative returns)
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf")  # No losses
        
        downside_std = np.std(downside)
        if downside_std == 0:
            return 0
        
        sortino = (avg_return - self.risk_free_rate / 252) / downside_std * np.sqrt(252)
        
        return sortino
    
    def _calculate_max_drawdown(self, profits: np.ndarray) -> float:
        """Maximum drawdown from peak."""
        if len(profits) == 0:
            return 0
        
        cumulative = np.cumsum(profits)
        peak = np.maximum.accumulate(cumulative)
        
        # Avoid div by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdown = np.where(peak > 0, (peak - cumulative) / peak, 0)
        
        return np.nanmax(drawdown)
    
    def _calculate_clv(
        self,
        bet_history: List[Dict],
        closing_odds: Dict[str, float],
    ) -> float:
        """
        Closing Line Value: how much better we got vs closing line.
        
        CLV > 0 indicates edge (got better odds than closing).
        """
        if not closing_odds:
            return 0
        
        total_edge = 0
        count = 0
        
        for bet in bet_history:
            match_id = bet.get("match_id")
            if match_id not in closing_odds:
                continue
            
            bet_odds = bet.get("odds", 0)
            close_odds = closing_odds.get(match_id, 0)
            
            if close_odds > 0 and bet_odds > 0:
                clv = (bet_odds - close_odds) / close_odds
                total_edge += clv
                count += 1
        
        return total_edge / count if count > 0 else 0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Skewness of returns distribution."""
        if len(returns) < 3:
            return 0
        
        mean = returns.mean()
        std = returns.std()
        
        if std == 0:
            return 0
        
        skew = ((returns - mean) ** 3).mean() / (std ** 3)
        return skew
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Kurtosis of returns (excess kurtosis, 0 = normal)."""
        if len(returns) < 4:
            return 0
        
        mean = returns.mean()
        std = returns.std()
        
        if std == 0:
            return 0
        
        kurt = ((returns - mean) ** 4).mean() / (std ** 4) - 3
        return kurt
    
    def _empty_metrics(self) -> MetricsSummary:
        """Return empty metrics."""
        return MetricsSummary(
            roi=0, win_rate=0, avg_odds=0, avg_ev=0,
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0, max_drawdown=0,
            yield_pct=0, clv=0, median_return=0, return_std=0,
            skewness=0, kurtosis=0,
        )
    
    def per_league_metrics(
        self,
        bet_history: List[Dict],
    ) -> Dict[str, MetricsSummary]:
        """Calculate metrics per league."""
        leagues = {}
        
        for bet in bet_history:
            league = bet.get("league", "Unknown")
            if league not in leagues:
                leagues[league] = []
            leagues[league].append(bet)
        
        return {
            league: self.calculate_all(bets)
            for league, bets in leagues.items()
        }
    
    def per_outcome_metrics(
        self,
        bet_history: List[Dict],
    ) -> Dict[str, Dict[str, float]]:
        """Calculate metrics per outcome type (home/draw/away)."""
        outcomes = {}
        
        for bet in bet_history:
            outcome_type = bet.get("selection", "unknown")
            if outcome_type not in outcomes:
                outcomes[outcome_type] = []
            outcomes[outcome_type].append(bet)
        
        result = {}
        for outcome_type, bets in outcomes.items():
            profits = [b.get("profit", 0) for b in bets]
            stakes = [b.get("stake", 1) for b in bets]
            wins = sum(1 for b in bets if b.get("outcome") == "win")
            
            total_profit = sum(profits)
            total_stake = sum(stakes)
            
            result[outcome_type] = {
                "count": len(bets),
                "wins": wins,
                "win_rate": wins / len(bets) if bets else 0,
                "roi": total_profit / total_stake if total_stake > 0 else 0,
                "profit": total_profit,
            }
        
        return result
