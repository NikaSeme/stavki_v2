"""
Betting Filters (Guardrails)
============================

Pre-bet and post-bet filters to avoid bad bets:
1. League-specific ROI cutoffs
2. Time-based filters (avoid early/late bets)
3. Odds movement filters
4. Correlation filters (avoid correlated bets)

All thresholds are data-driven, not hardcoded.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging

from .ev import EVResult

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filter application."""
    passed: bool
    reason: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


class BetFilters:
    """
    Comprehensive bet filtering system.
    
    All thresholds here should be optimized through backtesting.
    """
    
    # Default thresholds - TO BE OPTIMIZED
    DEFAULT_FILTERS = {
        # EV filters
        "min_ev": 0.03,                # 3% minimum EV
        "max_ev": 0.50,                # Cap crazy high EV (likely error)
        
        # Probability filters
        "min_prob": 0.10,
        "max_prob": 0.90,
        
        # Odds filters
        "min_odds": 1.20,
        "max_odds": 10.0,
        
        # Edge filters
        "min_edge": 0.02,              # 2% minimum edge vs market
        
        # Confidence filter
        "min_confidence": 0.05,        # Minimum model confidence
        
        # League filters (optimized per league)
        "league_min_roi": {},          # {league: min_roi} - disabled by default
        "excluded_leagues": set(),     # Leagues with persistent negative ROI
        
        # Time filters
        "min_hours_before_match": 1.0,  # Don't bet within 1 hour of kickoff
        "max_hours_before_match": 72.0, # Don't bet too early
        
        # Odds movement
        "max_odds_drift": 0.10,        # Max 10% odds change
        
        # Correlation
        "max_same_match_bets": 2,      # Max bets on same match
        "min_time_between_similar": 0, # Minutes between similar bets
    }
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        league_roi_history: Optional[Dict[str, float]] = None,
    ):
        self.config = {**self.DEFAULT_FILTERS, **(config or {})}
        
        # Historical ROI by league (for adaptive filtering)
        self.league_roi_history = league_roi_history or {}
        
        # Recent bets for correlation tracking
        self.recent_bets: List[Dict] = []
    
    def apply_all_filters(
        self,
        ev_result: EVResult,
        league: str = "unknown",
        match_datetime: Optional[datetime] = None,
        opening_odds: Optional[float] = None,
        model_confidence: Optional[float] = None,
    ) -> FilterResult:
        """
        Apply all configured filters to a bet.
        
        Returns:
            FilterResult with pass/fail and reason
        """
        # 1. EV filter
        if ev_result.ev < self.config["min_ev"]:
            return FilterResult(False, f"EV too low: {ev_result.ev:.2%}")
        
        if ev_result.ev > self.config["max_ev"]:
            return FilterResult(False, f"EV suspiciously high: {ev_result.ev:.2%}")
        
        # 2. Probability filter
        if not (self.config["min_prob"] <= ev_result.model_prob <= self.config["max_prob"]):
            return FilterResult(False, f"Prob outside range: {ev_result.model_prob:.2%}")
        
        # 3. Odds filter
        if not (self.config["min_odds"] <= ev_result.odds <= self.config["max_odds"]):
            return FilterResult(False, f"Odds outside range: {ev_result.odds}")
        
        # 4. Edge filter
        if ev_result.edge_pct < self.config["min_edge"]:
            return FilterResult(False, f"Edge too low: {ev_result.edge_pct:.2%}")
        
        # 5. Confidence filter
        if model_confidence is not None:
            if model_confidence < self.config["min_confidence"]:
                return FilterResult(False, f"Low confidence: {model_confidence:.2%}")
        
        # 6. League filter
        if league in self.config["excluded_leagues"]:
            return FilterResult(False, f"Excluded league: {league}")
        
        if league in self.config["league_min_roi"]:
            required_roi = self.config["league_min_roi"][league]
            if self.league_roi_history.get(league, 0) < required_roi:
                return FilterResult(False, f"League ROI below threshold: {league}")
        
        # 7. Time filter
        if match_datetime:
            hours_until = (match_datetime - datetime.now()).total_seconds() / 3600
            
            if hours_until < self.config["min_hours_before_match"]:
                return FilterResult(False, f"Too close to kickoff: {hours_until:.1f}h")
            
            if hours_until > self.config["max_hours_before_match"]:
                return FilterResult(False, f"Too far from kickoff: {hours_until:.1f}h")
        
        # 8. Odds drift filter
        if opening_odds is not None:
            drift = abs(ev_result.odds - opening_odds) / opening_odds
            if drift > self.config["max_odds_drift"]:
                return FilterResult(False, f"Odds drifted too much: {drift:.1%}")
        
        # 9. Correlation filter
        correlation_check = self._check_correlation(ev_result)
        if not correlation_check.passed:
            return correlation_check
        
        return FilterResult(True, details={
            "ev": ev_result.ev,
            "edge": ev_result.edge_pct,
            "odds": ev_result.odds,
        })
    
    def _check_correlation(self, ev_result: EVResult) -> FilterResult:
        """Check for correlated bets."""
        same_match_count = sum(
            1 for b in self.recent_bets 
            if b.get("match_id") == ev_result.match_id
        )
        
        if same_match_count >= self.config["max_same_match_bets"]:
            return FilterResult(False, f"Too many bets on same match: {same_match_count}")
        
        return FilterResult(True)
    
    def record_bet(self, ev_result: EVResult, league: str = "unknown"):
        """Record a bet for correlation tracking."""
        self.recent_bets.append({
            "match_id": ev_result.match_id,
            "market": ev_result.market,
            "selection": ev_result.selection,
            "timestamp": datetime.now().isoformat(),
            "league": league,
        })
        
        # Keep only last 100
        self.recent_bets = self.recent_bets[-100:]
    
    def update_league_roi(self, league: str, roi: float):
        """Update historical ROI for a league."""
        self.league_roi_history[league] = roi
    
    def auto_exclude_league(self, league: str, min_bets: int = 50, roi_threshold: float = -0.10):
        """Auto-exclude leagues with persistently negative ROI."""
        if self.league_roi_history.get(league, 0) < roi_threshold:
            self.config["excluded_leagues"].add(league)
            logger.warning(f"Auto-excluded league: {league}")
    
    def optimize_thresholds(
        self,
        historical_bets: List[Dict],
        target_metric: str = "roi",
    ) -> Dict[str, Any]:
        """
        Optimize filter thresholds based on historical data.
        
        THIS IS THE KEY FUNCTION - thresholds are data-driven.
        
        Args:
            historical_bets: List of historical bets with outcomes
            target_metric: "roi", "sharpe", or "profit"
        
        Returns:
            Optimized configuration
        """
        best_config = self.config.copy()
        best_score = float("-inf")
        
        # Grid search over key parameters
        min_ev_options = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10]
        min_edge_options = [0.01, 0.02, 0.03, 0.05]
        
        for min_ev in min_ev_options:
            for min_edge in min_edge_options:
                # Apply filters
                filtered = []
                for bet in historical_bets:
                    if bet.get("ev", 0) >= min_ev and bet.get("edge", 0) >= min_edge:
                        filtered.append(bet)
                
                if len(filtered) < 20:  # Minimum sample
                    continue
                
                # Calculate score
                total_staked = sum(b.get("stake", 1) for b in filtered)
                total_profit = sum(
                    b.get("stake", 1) * (b["odds"] - 1) if b.get("result") == "win" else -b.get("stake", 1)
                    for b in filtered if b.get("result") in ("win", "loss")
                )
                
                roi = total_profit / total_staked if total_staked > 0 else 0
                
                # Score (prefer higher ROI with reasonable bet count)
                n_bets = len(filtered)
                score = roi * min(n_bets / 100, 1)  # Penalize too few bets
                
                if score > best_score:
                    best_score = score
                    best_config["min_ev"] = min_ev
                    best_config["min_edge"] = min_edge
        
        logger.info(
            f"Optimized thresholds: min_ev={best_config['min_ev']:.2%}, "
            f"min_edge={best_config['min_edge']:.2%}"
        )
        
        return best_config


class MetaFilter:
    """
    Meta-filter combining multiple models' agreement.
    
    Only bets when models agree (reduces variance).
    """
    
    def __init__(
        self,
        min_models_agree: int = 2,
        max_disagreement: float = 0.15,  # Max prob difference between models
    ):
        self.min_models_agree = min_models_agree
        self.max_disagreement = max_disagreement
    
    def check_agreement(
        self,
        predictions: Dict[str, Dict[str, float]],  # model_name -> {outcome: prob}
        selection: str,
    ) -> FilterResult:
        """
        Check if enough models agree on the selection.
        
        Args:
            predictions: Dict of model predictions
            selection: Outcome to check (e.g., "home")
        
        Returns:
            FilterResult
        """
        if len(predictions) < self.min_models_agree:
            return FilterResult(False, "Not enough models")
        
        # Collect probabilities for selection
        probs = []
        best_selections = []
        
        for model_name, model_probs in predictions.items():
            probs.append(model_probs.get(selection, 0))
            best = max(model_probs.items(), key=lambda x: x[1])[0]
            best_selections.append(best)
        
        # Check agreement on selection
        agree_count = best_selections.count(selection)
        
        if agree_count < self.min_models_agree:
            return FilterResult(
                False, 
                f"Only {agree_count}/{len(predictions)} models agree"
            )
        
        # Check disagreement (variance)
        if len(probs) >= 2:
            std = np.std(probs)
            if std > self.max_disagreement:
                return FilterResult(
                    False,
                    f"Model disagreement too high: {std:.2%}"
                )
        
        return FilterResult(True, details={
            "agree_count": agree_count,
            "avg_prob": np.mean(probs),
            "std_prob": np.std(probs) if len(probs) >= 2 else 0,
        })
