"""
Expected Value (EV) Calculator
==============================

The core formula: EV = probability * odds - 1

Positive EV indicates a profitable bet in the long run.
This module provides EV calculation with market edge analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class EVResult:
    """EV calculation result for a single bet."""
    match_id: str
    market: str
    selection: str
    model_prob: float
    odds: float
    ev: float
    edge_pct: float  # Edge = model_prob - implied_prob
    implied_prob: float
    bookmaker: Optional[str] = None
    is_suspicious: bool = False
    
    @property
    def is_value(self) -> bool:
        return self.ev > 0
    
    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "market": self.market,
            "selection": self.selection,
            "model_prob": self.model_prob,
            "odds": self.odds,
            "ev": self.ev,
            "edge_pct": self.edge_pct,
            "implied_prob": self.implied_prob,
            "bookmaker": self.bookmaker,
            "is_suspicious": self.is_suspicious,
        }


class EVCalculator:
    """
    Expected Value calculator with multi-market support.
    """
    
    def __init__(
        self,
        min_ev: float = 0.03,       # 3% minimum EV
        min_prob: float = 0.10,      # 10% minimum model probability
        max_prob: float = 0.90,      # 90% maximum probability (avoid heavy favorites)
        min_odds: float = 1.20,      # Minimum odds
        max_odds: float = 10.0,      # Maximum odds
        use_no_vig: bool = True,     # Use no-vig odds for edge calculation
    ):
        self.min_ev = min_ev
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.use_no_vig = use_no_vig
    
    @staticmethod
    def calculate_ev(prob: float, odds: float) -> float:
        """
        Basic EV formula: EV = p * odds - 1
        
        Args:
            prob: Model probability (0-1)
            odds: Decimal odds (e.g., 2.5)
        
        Returns:
            Expected value (-1 to infinity)
        """
        if prob <= 0 or prob >= 1:
            return -1.0
        if odds <= 1.0:
            return -1.0
        return prob * odds - 1.0
    
    @staticmethod
    def implied_prob(odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 1.0:
            return 0.0
        return 1.0 / odds
    
    @staticmethod
    def remove_vig(odds_dict: Dict[str, float]) -> Dict[str, float]:
        """
        Remove bookmaker margin (vig) from odds.
        
        Uses proportional method: fair_prob = implied_prob / total_implied
        
        Args:
            odds_dict: {outcome: decimal_odds}
        
        Returns:
            {outcome: fair_probability}
        """
        implied = {k: 1.0/v for k, v in odds_dict.items() if v > 1}
        total = sum(implied.values())
        
        if total == 0:
            return {}
        
        return {k: v/total for k, v in implied.items()}
    
    def evaluate_bet(
        self,
        match_id: str,
        market: str,
        selection: str,
        model_prob: float,
        odds: float,
        bookmaker: Optional[str] = None,
        all_odds: Optional[Dict[str, float]] = None,  # For vig removal
    ) -> Optional[EVResult]:
        """
        Evaluate a single betting opportunity.
        
        Args:
            match_id: Match identifier
            market: Market type (1X2, O/U, BTTS, etc.)
            selection: Selection (home, draw, away, over_2.5, etc.)
            model_prob: Model's probability for this outcome
            odds: Available decimal odds
            bookmaker: Bookmaker name (optional)
            all_odds: All market odds for vig removal (optional)
        
        Returns:
            EVResult if passes filters, None otherwise
        """
        # Validate inputs
        if not (self.min_prob <= model_prob <= self.max_prob):
            return None
        
        if not (self.min_odds <= odds <= self.max_odds):
            return None
        
        # Calculate implied probability
        if self.use_no_vig and all_odds:
            fair_probs = self.remove_vig(all_odds)
            implied = fair_probs.get(selection, 1/odds)
        else:
            implied = 1/odds
        
        # Calculate EV
        ev = self.calculate_ev(model_prob, odds)
        
        # Calculate edge
        edge = model_prob - implied
        
        # Apply filters
        if ev < self.min_ev:
            return None
            
        # Optimization: Flag suspicious high EV bets (Data integrity check)
        # Block hallucinations completely (EV > 80%)
        is_suspicious = False
        if ev > 0.80:
            logger.warning(
                f"🚨 BLOCKED: Mathematically impossible EV ({ev:.2%}) for {match_id} ({selection}). "
                f"Odds: {odds}, Prob: {model_prob:.2%} | This usually means missing features or API mapping drops."
            )
            return None
        elif ev > 0.50:
            logger.warning(
                f"⚠️ Suspicious High EV ({ev:.2%}) for {match_id} ({selection}). "
                f"Odds: {odds}, Prob: {model_prob:.2%} | Keeping but flagging."
            )
            is_suspicious = True
        
        return EVResult(
            match_id=match_id,
            market=market,
            selection=selection,
            model_prob=model_prob,
            odds=odds,
            ev=ev,
            edge_pct=edge,
            implied_prob=implied,
            bookmaker=bookmaker,
            is_suspicious=is_suspicious,
        )
    
    def find_value_bets(
        self,
        predictions: List[Dict],
        odds_data: pd.DataFrame,
    ) -> List[EVResult]:
        """
        Find all value bets from predictions and odds.
        
        Args:
            predictions: List of model predictions
            odds_data: DataFrame with odds (HomeTeam, AwayTeam, odds columns)
        
        Returns:
            List of EVResult sorted by EV descending
        """
        value_bets = []
        
        for pred in predictions:
            match_id = pred.get("match_id")
            market = pred.get("market", "1x2")
            probabilities = pred.get("probabilities", {})
            
            # Find matching odds
            odds_row = self._find_odds(odds_data, match_id)
            
            if odds_row is None:
                continue
            
            # Check each outcome
            for selection, prob in probabilities.items():
                odds_value = self._get_odds_for_selection(odds_row, market, selection)
                
                if odds_value is None:
                    continue
                
                # Get all odds for vig removal
                all_odds = self._get_all_market_odds(odds_row, market)
                
                result = self.evaluate_bet(
                    match_id=match_id,
                    market=market,
                    selection=selection,
                    model_prob=prob,
                    odds=odds_value,
                    bookmaker=odds_row.get("Bookmaker"),
                    all_odds=all_odds,
                )
                
                if result:
                    value_bets.append(result)
        
        # Sort by EV
        value_bets.sort(key=lambda x: -x.ev)
        
        logger.info(f"Found {len(value_bets)} value bets from {len(predictions)} predictions")
        
        return value_bets
    
    def _build_odds_index(self, odds_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Pre-build {match_id: row} dict for O(1) odds lookup."""
        index = {}
        for idx, row in odds_data.iterrows():
            potential_id = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}")
            index[str(potential_id)] = row
        return index
    
    def _find_odds(
        self, 
        odds_data: pd.DataFrame, 
        match_id: str,
        _cache: Dict[str, pd.Series] = None,
    ) -> Optional[pd.Series]:
        """Find odds row for a match using pre-built index."""
        if _cache is not None:
            return _cache.get(match_id)
        
        # Fallback: build on-the-fly (slower)
        for idx, row in odds_data.iterrows():
            potential_id = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}")
            if str(potential_id) == match_id:
                return row
        return None
    
    def _get_odds_for_selection(
        self, 
        row: pd.Series, 
        market: str, 
        selection: str
    ) -> Optional[float]:
        """Extract odds for a specific selection."""
        # 1X2 market
        if market in ("1x2", "match_winner"):
            if selection == "home":
                return row.get("AvgH") or row.get("PSH") or row.get("B365H") or row.get("Odds_1X2_Home")
            elif selection == "draw":
                return row.get("AvgD") or row.get("PSD") or row.get("B365D") or row.get("Odds_1X2_Draw")
            elif selection == "away":
                return row.get("AvgA") or row.get("PSA") or row.get("B365A") or row.get("Odds_1X2_Away")
        
        # O/U market
        elif market == "over_under":
            if "over" in selection:
                return row.get("B365>2.5") or row.get("Odds_OU_Over")
            elif "under" in selection:
                return row.get("B365<2.5") or row.get("Odds_OU_Under")
        
        # BTTS market
        elif market == "btts":
            if selection == "yes":
                return row.get("BTTS_Yes") or row.get("Odds_BTTS_Yes")
            elif selection == "no":
                return row.get("BTTS_No") or row.get("Odds_BTTS_No")
        
        return None
    
    def _get_all_market_odds(
        self, 
        row: pd.Series, 
        market: str
    ) -> Dict[str, float]:
        """Get all odds for a market (for vig removal)."""
        odds = {}
        
        if market in ("1x2", "match_winner"):
            h = row.get("AvgH") or row.get("PSH") or row.get("B365H")
            d = row.get("AvgD") or row.get("PSD") or row.get("B365D")
            a = row.get("AvgA") or row.get("PSA") or row.get("B365A")
            
            if h: odds["home"] = h
            if d: odds["draw"] = d
            if a: odds["away"] = a
        
        elif market == "over_under":
            over = row.get("B365>2.5")
            under = row.get("B365<2.5")
            
            if over: odds["over_2.5"] = over
            if under: odds["under_2.5"] = under
        
        return odds


def compute_ev(prob: float, odds: float) -> float:
    """Simple EV calculation (standalone function)."""
    return EVCalculator.calculate_ev(prob, odds)


def filter_positive_ev(
    predictions_df: pd.DataFrame,
    ev_threshold: float = 0.03,
    prob_column: str = "prob_home",
    odds_column: str = "odds_home",
) -> pd.DataFrame:
    """
    Filter DataFrame to positive EV bets only.
    
    Args:
        predictions_df: DataFrame with probabilities and odds
        ev_threshold: Minimum EV
        prob_column: Column with model probability
        odds_column: Column with odds
    
    Returns:
        Filtered DataFrame with 'ev' column added
    """
    df = predictions_df.copy()
    
    df["ev"] = df.apply(
        lambda r: compute_ev(r[prob_column], r[odds_column]),
        axis=1
    )
    
    filtered = df[df["ev"] >= ev_threshold].copy()
    
    logger.info(f"Filtered {len(filtered)}/{len(df)} bets with EV >= {ev_threshold:.2%}")
    
    return filtered
