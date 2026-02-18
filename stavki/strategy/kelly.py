"""
Kelly Criterion Staking with Risk Management
=============================================

Implements:
1. Fractional Kelly (safer than full Kelly)
2. Exposure limits (per bet, per league, per day)
3. Drawdown protection
4. Data-driven parameter optimization

Key Insight: Kelly fraction should be optimized per dataset, not hardcoded.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import json
import logging

from .ev import EVResult, EVCalculator

logger = logging.getLogger(__name__)


@dataclass
class StakeResult:
    """Result of stake calculation."""
    match_id: str
    market: str
    selection: str
    stake_pct: float      # Fraction of bankroll
    stake_amount: float   # Absolute stake
    kelly_full: float     # Full Kelly (for reference)
    ev: float
    odds: float
    model_prob: float
    reason: Optional[str] = None  # If stake=0, why
    
    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "market": self.market,
            "selection": self.selection,
            "stake_pct": self.stake_pct,
            "stake_amount": self.stake_amount,
            "kelly_full": self.kelly_full,
            "ev": self.ev,
            "odds": self.odds,
            "model_prob": self.model_prob,
            "reason": self.reason,
        }


class KellyStaker:
    """
    Kelly criterion staking with comprehensive risk management.
    
    All parameters are optimizable through backtesting, not hardcoded.
    """
    
    # Default config - THESE ARE OPTIMIZED through backtesting
    DEFAULT_CONFIG = {
        # Kelly parameters
        "kelly_fraction": 0.75,           # BOB'S AGGRESSIVE: 75% Kelly
        
        # Stake limits
        "max_stake_pct": 0.05,            # Max single bet: 5%
        "min_stake_pct": 0.001,           # Min stake: 0.1%
        "min_stake_amount": 1.0,          # Minimum stake in currency
        
        # Exposure limits
        "max_daily_exposure_pct": 0.20,   # Max 20% bankroll per day
        "max_league_exposure_pct": 0.10,  # Max 10% per league
        "max_concurrent_bets": 50,        # Max open bets at once
        
        # Drawdown limits
        "drawdown_reduce_threshold": 0.15,  # Reduce stakes at 15% drawdown
        "drawdown_pause_threshold": 0.25,   # Pause at 25% drawdown
        "drawdown_reduction_factor": 0.5,   # Halve stakes during drawdown
    }
    
    def __init__(
        self,
        bankroll: float = 1000.0,
        config: Optional[Dict[str, float]] = None,
        state_file: Optional[str] = None,
    ):
        self.initial_bankroll = bankroll
        self.bankroll = bankroll
        self.peak_bankroll = bankroll
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.state_file = Path(state_file) if state_file else None
        
        # Tracking
        self.bet_history: List[Dict] = []
        self.pending_bets: List[Dict] = []
        self.daily_exposure: Dict[str, float] = defaultdict(float)
        self.league_exposure: Dict[str, float] = defaultdict(float)
        
        # Load state if available
        if self.state_file and self.state_file.exists():
            self._load_state()
    
    @staticmethod
    def kelly_formula(prob: float, odds: float) -> float:
        """
        Calculate full Kelly fraction.
        
        Formula: f* = (b*p - q) / b
        where b = odds - 1, p = prob, q = 1 - p
        
        Returns:
            Fraction of bankroll to bet (0 to 1, can be > 1 for high edge)
        """
        if prob <= 0 or prob >= 1 or odds <= 1:
            return 0.0
        
        b = odds - 1
        p = prob
        q = 1 - p
        
        kelly = (b * p - q) / b
        
        return max(0.0, kelly)
    
    def calculate_stake(
        self,
        ev_result: EVResult,
        league: str = "unknown",
        date: Optional[datetime] = None,
        apply_limits: bool = True,
    ) -> StakeResult:

        """
        Calculate stake for a betting opportunity.
        
        Args:
            ev_result: EV calculation result
            league: League code for exposure tracking
            apply_limits: Whether to apply risk limits
        
        Returns:
            StakeResult with recommended stake
        """
        # Type safety check
        if not isinstance(ev_result, EVResult):
            raise TypeError(
                f"Expected EVResult object, got {type(ev_result).__name__}. "
                "Did you pass raw floats? Wrap them in EVResult(...) first."
            )

        prob = ev_result.model_prob
        odds = ev_result.odds
        
        # Calculate full Kelly
        kelly_full = self.kelly_formula(prob, odds)
        
        if kelly_full <= 0:
            return StakeResult(
                match_id=ev_result.match_id,
                market=ev_result.market,
                selection=ev_result.selection,
                stake_pct=0.0,
                stake_amount=0.0,
                kelly_full=kelly_full,
                ev=ev_result.ev,
                odds=odds,
                model_prob=prob,
                reason="Negative Kelly",
            )
        
        # Apply fractional Kelly
        stake_pct = kelly_full * self.config["kelly_fraction"]
        
        if not apply_limits:
            stake_amount = self.bankroll * stake_pct
            return StakeResult(
                match_id=ev_result.match_id,
                market=ev_result.market,
                selection=ev_result.selection,
                stake_pct=stake_pct,
                stake_amount=stake_amount,
                kelly_full=kelly_full,
                ev=ev_result.ev,
                odds=odds,
                model_prob=prob,
            )
        
        # Apply limits
        stake_pct, reason = self._apply_all_limits(stake_pct, league, date)

        
        if stake_pct <= 0:
            return StakeResult(
                match_id=ev_result.match_id,
                market=ev_result.market,
                selection=ev_result.selection,
                stake_pct=0.0,
                stake_amount=0.0,
                kelly_full=kelly_full,
                ev=ev_result.ev,
                odds=odds,
                model_prob=prob,
                reason=reason,
            )
        
        stake_amount = self.bankroll * stake_pct
        
        # Check minimum stake
        if stake_amount < self.config["min_stake_amount"]:
            return StakeResult(
                match_id=ev_result.match_id,
                market=ev_result.market,
                selection=ev_result.selection,
                stake_pct=0.0,
                stake_amount=0.0,
                kelly_full=kelly_full,
                ev=ev_result.ev,
                odds=odds,
                model_prob=prob,
                reason="Below minimum stake",
            )
        
        return StakeResult(
            match_id=ev_result.match_id,
            market=ev_result.market,
            selection=ev_result.selection,
            stake_pct=stake_pct,
            stake_amount=round(stake_amount, 2),
            kelly_full=kelly_full,
            ev=ev_result.ev,
            odds=odds,
            model_prob=prob,
        )
    
    def _apply_all_limits(
        self, 
        stake_pct: float, 
        league: str,
        date: Optional[datetime] = None,
    ) -> Tuple[float, Optional[str]]:
        """Apply all stake limits."""
        if date:
            today = date.strftime("%Y-%m-%d")
        else:
            today = datetime.now().strftime("%Y-%m-%d")
        
        # 1. Max stake per bet
        if stake_pct > self.config["max_stake_pct"]:
            stake_pct = self.config["max_stake_pct"]
        
        # 2. Check drawdown pause
        current_drawdown = self._get_current_drawdown()
        
        if current_drawdown >= self.config["drawdown_pause_threshold"]:
            return 0.0, f"Drawdown pause ({current_drawdown:.1%})"
        
        # 3. Apply drawdown reduction
        if current_drawdown >= self.config["drawdown_reduce_threshold"]:
            stake_pct *= self.config["drawdown_reduction_factor"]
        
        # 4. Daily exposure limit
        daily_used = self.daily_exposure[today]
        daily_remaining = self.config["max_daily_exposure_pct"] - daily_used
        
        if daily_remaining <= 0:
            return 0.0, "Daily exposure limit"
        
        stake_pct = min(stake_pct, daily_remaining)
        
        # 5. League exposure limit
        league_used = self.league_exposure[league]
        league_remaining = self.config["max_league_exposure_pct"] - league_used
        
        if league_remaining <= 0:
            return 0.0, f"League exposure limit ({league})"
        
        stake_pct = min(stake_pct, league_remaining)
        
        # 6. Max concurrent bets
        if len(self.pending_bets) >= self.config["max_concurrent_bets"]:
            return 0.0, "Max concurrent bets"
        
        # 7. Minimum stake
        if stake_pct < self.config["min_stake_pct"]:
            return 0.0, "Below minimum stake percentage"
        
        return stake_pct, None
    
    def _get_current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_bankroll <= 0:
            return 0.0
        
        return (self.peak_bankroll - self.bankroll) / self.peak_bankroll
    
    def place_bet(
        self, 
        stake_result: StakeResult, 
        league: str = "unknown",
        date: Optional[datetime] = None,
    ):
        """Record a placed bet (pending until settled)."""
        if stake_result.stake_amount <= 0:
            return
        
        if date:
            today = date.strftime("%Y-%m-%d")
            timestamp = date.isoformat()
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().isoformat()
        
        bet_record = {
            "timestamp": timestamp,
            "match_id": stake_result.match_id,
            "market": stake_result.market,
            "selection": stake_result.selection,
            "stake": stake_result.stake_amount,
            "odds": stake_result.odds,
            "ev": stake_result.ev,
            "league": league,
            "status": "pending",
        }
        
        self.pending_bets.append(bet_record)
        
        # Update exposure
        stake_pct = stake_result.stake_amount / self.bankroll
        self.daily_exposure[today] += stake_pct
        self.league_exposure[league] += stake_pct
        
        self._save_state()
    
    def settle_bet(
        self,
        match_id: str,
        result: str,  # "win", "loss", "void", "half_win", "half_loss"
        date: Optional[datetime] = None,
    ):

        """Settle a pending bet."""
        # Find the bet
        bet_idx = None
        for i, bet in enumerate(self.pending_bets):
            if bet["match_id"] == match_id:
                bet_idx = i
                break
        
        if bet_idx is None:
            logger.warning(f"Bet not found: {match_id}")
            return
        
        bet = self.pending_bets.pop(bet_idx)
        stake = bet["stake"]
        odds = bet["odds"]
        league = bet["league"]
        
        # Calculate profit
        if result == "win":
            profit = stake * (odds - 1)
        elif result == "loss":
            profit = -stake
        elif result == "half_win":
            profit = stake * (odds - 1) * 0.5
        elif result == "half_loss":
            profit = -stake * 0.5
        else:  # void
            profit = 0.0
        
        # Update bankroll
        self.bankroll += profit
        
        # Update peak
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        
        # Release exposure
        stake_pct = stake / (self.bankroll - profit) if (self.bankroll - profit) > 0 else 0
        
        if date:
            today = date.strftime("%Y-%m-%d")
            settled_at = date.isoformat()
        else:
            today = datetime.now().strftime("%Y-%m-%d")
            settled_at = datetime.now().isoformat()
            
        self.daily_exposure[today] = max(0, self.daily_exposure[today] - stake_pct)
        self.league_exposure[league] = max(0, self.league_exposure[league] - stake_pct)
        
        # Record
        bet["status"] = result
        bet["profit"] = profit
        bet["settled_at"] = settled_at
        bet["bankroll_after"] = self.bankroll
        self.bet_history.append(bet)
        
        self._save_state()
        
        logger.info(
            f"Settled: {result} | Stake: {stake:.2f} | "
            f"Profit: {profit:+.2f} | Bankroll: {self.bankroll:.2f}"
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get betting statistics."""
        if not self.bet_history:
            return {
                "total_bets": 0,
                "pending_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_staked": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
                "bankroll": self.bankroll,
                "peak_bankroll": self.peak_bankroll,
                "drawdown": 0.0,
                "initial_bankroll": self.initial_bankroll,
            }

        
        wins = sum(1 for b in self.bet_history if b["status"] == "win")
        losses = sum(1 for b in self.bet_history if b["status"] == "loss")
        settled = [b for b in self.bet_history if b["status"] in ("win", "loss")]
        
        total_staked = sum(b["stake"] for b in settled)
        total_profit = sum(b["profit"] for b in settled)
        
        return {
            "total_bets": len(self.bet_history),
            "pending_bets": len(self.pending_bets),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(settled) if settled else 0,
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": total_profit / total_staked if total_staked > 0 else 0,
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "drawdown": self._get_current_drawdown(),
            "initial_bankroll": self.initial_bankroll,
        }
    
    def optimize_kelly_fraction(
        self,
        historical_bets: List[Dict],
        fractions: List[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize Kelly fraction on historical data using vectorized simulation.
        
        Args:
            historical_bets: List of historical bets with prob, odds, result
            fractions: Kelly fractions to test (default: 0.05 to 0.50)
        
        Returns:
            (optimal_fraction, {fraction: final_roi})
        """
        if not historical_bets:
            return 0.25, {}
            
        if fractions is None:
            fractions = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        
        # 1. Convert to arrays for vectorization
        probs = np.array([b.get("model_prob", 0) for b in historical_bets])
        odds = np.array([b.get("odds", 0) for b in historical_bets])
        
        # Outcome: 1 for win, 0 for loss/void (voids should ideally be handled)
        # Assuming simple win/loss for now or status check
        results = np.array([1.0 if b.get("result") == "win" or b.get("status") == "win" else 0.0 for b in historical_bets])
        
        # 2. Pre-calculate Full Kelly for all bets
        # Kelly = (b*p - q) / b
        b_odds = odds - 1
        q_probs = 1 - probs
        kelly_full = np.divide(
            (b_odds * probs - q_probs), 
            b_odds, 
            out=np.zeros_like(probs), 
            where=b_odds!=0
        )
        kelly_full = np.maximum(kelly_full, 0)
        
        results_map = {}
        bankroll_start = 1000.0
        
        for fraction in fractions:
            # 3. Apply fraction and limits
            stake_pct = kelly_full * fraction
            stake_pct = np.minimum(stake_pct, self.config["max_stake_pct"])
            
            # 4. Calculate returns
            # If win: growth = 1 + stake_pct * (odds - 1)
            # If loss: growth = 1 - stake_pct
            
            # Vectorized profit multiplier per bet
            # profit_mult = (odds - 1) * results - (1 - results) * 1
            # But simpler:
            match_return = np.where(results == 1, (odds - 1), -1.0)
            
            # Bankroll multipliers: (1 + stake_pct * match_return)
            growth_factors = 1 + stake_pct * match_return
            
            # 5. Simulate trajectory (Cumulative Product)
            # Clip growth to 0 to represent bankruptcy (avoid negative bankroll math)
            growth_factors = np.maximum(growth_factors, 0)
            
            trajectory = bankroll_start * np.cumprod(growth_factors)
            
            if len(trajectory) == 0:
                continue
                
            final_bankroll = trajectory[-1]
            profit = final_bankroll - bankroll_start
            roi = profit / bankroll_start  # Simple ROI relative to starting bankroll
            
            # 6. Calculate Drawdown vectorized
            # Running max bankroll
            running_max = np.maximum.accumulate(trajectory)
            # Current drawdown at each step
            drawdowns = (running_max - trajectory) / running_max
            # Handle division by zero if running_max is 0 (bankruptcy)
            drawdowns = np.nan_to_num(drawdowns, 0.0)
            
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
            
            results_map[fraction] = {
                "final_bankroll": final_bankroll,
                "roi": roi,
                "max_drawdown": max_drawdown,
                "risk_adjusted": roi / max(max_drawdown, 0.01),
            }
            
        # Find best fraction
        if not results_map:
            return 0.25, {}
            
        best_fraction = max(results_map.keys(), key=lambda f: results_map[f]["risk_adjusted"])
        
        logger.info(
            f"Optimal Kelly fraction: {best_fraction:.2f} "
            f"(ROI: {results_map[best_fraction]['roi']:.2%})"
        )
        
        return best_fraction, results_map
    
    def _save_state(self):
        """Save state to file."""
        if not self.state_file:
            return
        
        state = {
            "bankroll": self.bankroll,
            "peak_bankroll": self.peak_bankroll,
            "initial_bankroll": self.initial_bankroll,
            "config": self.config,
            "daily_exposure": dict(self.daily_exposure),
            "league_exposure": dict(self.league_exposure),
            "pending_bets": self.pending_bets,
            "bet_history": self.bet_history[-500:],  # Keep last 500
            "last_updated": datetime.now().isoformat(),
        }
        
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self):
        """Load state from file."""
        try:
            with open(self.state_file) as f:
                state = json.load(f)
            
            self.bankroll = state.get("bankroll", self.bankroll)
            self.peak_bankroll = state.get("peak_bankroll", self.peak_bankroll)
            self.config = {**self.config, **state.get("config", {})}
            self.daily_exposure = defaultdict(float, state.get("daily_exposure", {}))
            self.league_exposure = defaultdict(float, state.get("league_exposure", {}))
            self.pending_bets = state.get("pending_bets", [])
            self.bet_history = state.get("bet_history", [])
            
            logger.info(f"Loaded state: bankroll={self.bankroll:.2f}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")


def kelly_simple(prob: float, odds: float, fraction: float = 0.25) -> float:
    """Simple Kelly stake calculation (standalone)."""
    return KellyStaker.kelly_formula(prob, odds) * fraction
