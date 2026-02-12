"""
Prediction and bet-related schemas.

These represent the outputs of our prediction pipeline
and the bets we place.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, computed_field


class BetStatus(str, Enum):
    """Bet lifecycle status."""
    PENDING = "pending"      # Signal generated, not yet placed
    PLACED = "placed"        # Bet submitted to bookmaker
    WON = "won"              # Settled as win
    LOST = "lost"            # Settled as loss
    VOID = "void"            # Cancelled/void
    PARTIAL = "partial"      # Partially matched (exchange)
    REJECTED = "rejected"    # Bookmaker rejected


class Prediction(BaseModel):
    """
    Model prediction for a single match.
    
    Contains probabilities from each model component
    plus the final ensemble prediction.
    """
    
    match_id: str
    timestamp: datetime
    
    # Component model probabilities [home, draw, away]
    poisson_probs: List[float] = Field(..., min_length=3, max_length=3)
    catboost_probs: List[float] = Field(..., min_length=3, max_length=3)
    neural_probs: List[float] = Field(..., min_length=3, max_length=3)
    
    # Final ensemble probabilities
    ensemble_probs: List[float] = Field(..., min_length=3, max_length=3)
    
    # Weights used
    ensemble_weights: Dict[str, float] = Field(default_factory=dict)
    
    # Calibration applied?
    is_calibrated: bool = False
    
    # Additional signals
    disagreement_score: float = 0.0  # How much models disagree
    confidence_score: float = 0.0     # Overall confidence
    
    @computed_field
    @property
    def home_prob(self) -> float:
        return self.ensemble_probs[0]
    
    @computed_field
    @property
    def draw_prob(self) -> float:
        return self.ensemble_probs[1]
    
    @computed_field
    @property
    def away_prob(self) -> float:
        return self.ensemble_probs[2]
    
    @computed_field
    @property
    def max_confidence(self) -> float:
        """Highest probability prediction."""
        return max(self.ensemble_probs)
    
    @computed_field
    @property
    def best_outcome(self) -> str:
        """Most likely outcome."""
        idx = self.ensemble_probs.index(max(self.ensemble_probs))
        return ["home", "draw", "away"][idx]
    
    def get_prob(self, outcome: str) -> float:
        """Get probability for specific outcome."""
        mapping = {"home": 0, "draw": 1, "away": 2}
        return self.ensemble_probs[mapping[outcome.lower()]]


class ValueSignal(BaseModel):
    """
    A detected value betting opportunity.
    
    Generated when our model probability significantly
    exceeds implied market probability.
    """
    
    match_id: str
    timestamp: datetime
    
    # The bet
    outcome: str  # "home", "draw", "away"
    odds: float
    bookmaker: str
    
    # Probabilities
    model_prob: float
    market_prob: float  # No-vig implied
    
    # Value metrics
    ev_pct: float  # Expected value percentage
    edge_pct: float  # Model prob - market prob
    kelly_pct: float  # Raw Kelly stake %
    
    # Confidence & filters
    confidence: float
    divergence_pct: float  # How far from market consensus
    
    # Guardrail flags
    is_outlier: bool = False
    bookmaker_count: int = 1
    min_bookmaker_odds: float = 0
    max_bookmaker_odds: float = 0
    
    # Meta-filter score (if applied)
    meta_filter_score: Optional[float] = None
    passes_meta_filter: bool = True
    
    # League context
    league: str = ""
    
    @computed_field
    @property
    def odds_spread(self) -> float:
        """Gap between min and max odds (outlier indicator)."""
        if self.max_bookmaker_odds > 0 and self.min_bookmaker_odds > 0:
            return self.max_bookmaker_odds - self.min_bookmaker_odds
        return 0
    
    @computed_field
    @property
    def signal_strength(self) -> str:
        """Categorize signal strength."""
        if self.ev_pct >= 0.15:
            return "strong"
        elif self.ev_pct >= 0.10:
            return "medium"
        elif self.ev_pct >= 0.05:
            return "weak"
        return "marginal"


class BetRecommendation(BaseModel):
    """
    Final bet recommendation after all filters and staking.
    
    This is what gets sent to the user or auto-placed.
    """
    
    # Source signal
    signal: ValueSignal
    
    # Staking
    recommended_stake_pct: float  # % of bankroll
    recommended_stake_amount: float  # In currency
    
    # Kelly adjustments
    kelly_fraction_used: float = 0.25
    stake_capped: bool = False  # Hit max stake limit
    
    # Risk context
    current_drawdown: float = 0
    exposure_after: float = 0  # Total exposure if bet placed
    
    # Priority for multiple bets
    priority: int = 1
    
    @computed_field
    @property
    def expected_profit(self) -> float:
        """Expected profit from this bet."""
        return self.recommended_stake_amount * self.signal.ev_pct


class PlacedBet(BaseModel):
    """
    Record of an actually placed bet.
    
    For tracking, P&L, and CLV analysis.
    """
    
    id: str = Field(..., description="Unique bet ID")
    match_id: str
    
    # What we bet on
    outcome: str
    odds_at_placement: float
    stake: float
    bookmaker: str
    
    # When
    placed_at: datetime
    
    # Model state at time of bet
    model_prob: float
    ev_at_placement: float
    
    # Status
    status: BetStatus = BetStatus.PENDING
    
    # Settlement
    settled_at: Optional[datetime] = None
    result: Optional[str] = None  # "win", "loss", "void"
    profit_loss: Optional[float] = None
    
    # CLV tracking
    closing_odds: Optional[float] = None
    clv_pct: Optional[float] = None  # Closing Line Value
    
    @computed_field
    @property
    def potential_return(self) -> float:
        """Potential return if bet wins."""
        return self.stake * self.odds_at_placement
    
    @computed_field
    @property
    def potential_profit(self) -> float:
        """Potential profit if bet wins."""
        return self.stake * (self.odds_at_placement - 1)
    
    def settle(self, won: bool, closing_odds: Optional[float] = None) -> None:
        """Settle the bet with result."""
        self.settled_at = datetime.utcnow()
        
        if won:
            self.status = BetStatus.WON
            self.result = "win"
            self.profit_loss = self.stake * (self.odds_at_placement - 1)
        else:
            self.status = BetStatus.LOST
            self.result = "loss"
            self.profit_loss = -self.stake
        
        if closing_odds:
            self.closing_odds = closing_odds
            # CLV = (our_odds / closing_odds - 1) * 100
            self.clv_pct = (self.odds_at_placement / closing_odds - 1) * 100


class DailyStats(BaseModel):
    """
    Daily betting statistics for monitoring.
    """
    
    date: str  # YYYY-MM-DD
    
    # Volume
    total_bets: int = 0
    total_staked: float = 0
    
    # Results
    wins: int = 0
    losses: int = 0
    voids: int = 0
    
    # P&L
    gross_profit: float = 0
    gross_loss: float = 0
    net_pnl: float = 0
    
    # Metrics
    roi_pct: float = 0
    avg_odds: float = 0
    avg_ev: float = 0
    avg_clv: float = 0
    
    @computed_field
    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        if total == 0:
            return 0
        return self.wins / total
    
    @computed_field
    @property
    def yield_pct(self) -> float:
        """Net profit per unit staked."""
        if self.total_staked == 0:
            return 0
        return self.net_pnl / self.total_staked * 100
