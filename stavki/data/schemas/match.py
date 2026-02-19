"""
Core data schemas for STAVKI v2.

Enhanced Pydantic models with:
- Strict validation
- Computed properties for derived values
- JSON serialization support
- Hashable for caching
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, computed_field
import hashlib
from numpy import dtype # Compatibility for unpickling old objects


class League(str, Enum):
    """Supported leagues with their API identifiers."""
    EPL = "soccer_epl"
    LA_LIGA = "soccer_spain_la_liga"
    BUNDESLIGA = "soccer_germany_bundesliga"
    SERIE_A = "soccer_italy_serie_a"
    LIGUE_1 = "soccer_france_ligue_one"
    CHAMPIONSHIP = "soccer_efl_champ"
    NBA = "basketball_nba"
    
    @property
    def is_football(self) -> bool:
        return self.value.startswith("soccer")
    
    @property
    def display_name(self) -> str:
        names = {
            "soccer_epl": "Premier League",
            "soccer_spain_la_liga": "La Liga",
            "soccer_germany_bundesliga": "Bundesliga",
            "soccer_italy_serie_a": "Serie A",
            "soccer_france_ligue_one": "Ligue 1",
            "soccer_efl_champ": "Championship",
            "basketball_nba": "NBA",
        }
        return names.get(self.value, self.value)


class Outcome(str, Enum):
    """Match outcome types."""
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"


class Team(BaseModel):
    """Team with normalization support."""
    
    name: str = Field(..., min_length=1)
    normalized_name: str = ""
    
    def model_post_init(self, __context: Any) -> None:
        if not self.normalized_name:
            # Use specific normalizer if available
            try:
                from stavki.utils.team_names import normalize_team_name
                object.__setattr__(self, 'normalized_name', normalize_team_name(self.name))
            except ImportError:
                # Fallback
                object.__setattr__(self, 'normalized_name', self.name.lower().strip())
    
    def __hash__(self) -> int:
        return int(hashlib.md5(self.normalized_name.encode()).hexdigest(), 16) % (2**61 - 1)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Team):
            return self.normalized_name == other.normalized_name
        return False


class Match(BaseModel):
    """
    Central match entity with all metadata needed for predictions.
    
    Enhancements:
    - Unique hash for deduplication
    - Time-to-kickoff calculation
    - Result storage for historical data
    """
    
    id: str = Field(..., description="Unique match identifier")
    home_team: Team
    away_team: Team
    league: League
    commence_time: datetime
    
    # Results (None for upcoming matches)
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    
    # Metadata
    season: Optional[str] = None
    matchday: Optional[int] = None
    venue: Optional[str] = None
    
    # API source tracking (for multi-source reconciliation)
    source: str = "unknown"
    source_id: Optional[str] = None
    
    # Rich Data (Phase 2)
    stats: Optional["MatchStats"] = None
    lineups: Optional["MatchLineups"] = None
    enrichment: Optional["MatchEnrichment"] = None
    
    @computed_field
    @property
    def is_completed(self) -> bool:
        """Match has a result."""
        return self.home_score is not None and self.away_score is not None
    
    @computed_field
    @property
    def result(self) -> Optional[Outcome]:
        """Match result if completed."""
        if not self.is_completed:
            return None
        if self.home_score > self.away_score:
            return Outcome.HOME
        elif self.home_score < self.away_score:
            return Outcome.AWAY
        else:
            return Outcome.DRAW
    
    @computed_field
    @property
    def total_goals(self) -> Optional[int]:
        """Total goals if completed."""
        if not self.is_completed:
            return None
        return self.home_score + self.away_score
    
    def hours_until_kickoff(self, now: Optional[datetime] = None) -> float:
        """Hours until match starts."""
        if now is None:
            # Match commence_time timezone-awareness
            if self.commence_time.tzinfo is not None:
                now = datetime.now(timezone.utc)
            else:
                now = datetime.utcnow()
        delta = self.commence_time - now
        return delta.total_seconds() / 3600
    
    @computed_field
    @property
    def match_hash(self) -> str:
        """Unique hash for deduplication across sources."""
        key = f"{self.home_team.normalized_name}_{self.away_team.normalized_name}_{self.commence_time.date()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def __hash__(self) -> int:
        return int(self.match_hash, 16) % (2**61 - 1)


class OddsSnapshot(BaseModel):
    """
    Single odds snapshot from one bookmaker at one point in time.
    
    Critical for:
    - CLV (Closing Line Value) tracking
    - Line movement detection
    - Sharp money signals
    """
    
    match_id: str
    bookmaker: str
    timestamp: datetime
    
    # Core odds
    home_odds: float = Field(..., gt=1.0)
    draw_odds: Optional[float] = Field(None, gt=1.0)  # None for basketball
    away_odds: float = Field(..., gt=1.0)
    
    # Market type
    market: str = "h2h"  # h2h, spreads, totals
    
    @field_validator('home_odds', 'away_odds', 'draw_odds', mode='before')
    @classmethod
    def round_odds(cls, v):
        if v is not None:
            return round(float(v), 3)
        return v
    
    @computed_field
    @property
    def overround(self) -> float:
        """Bookmaker margin (vig). Lower = sharper book."""
        probs = 1/self.home_odds + 1/self.away_odds
        if self.draw_odds:
            probs += 1/self.draw_odds
        return (probs - 1) * 100  # As percentage
    
    @computed_field
    @property
    def implied_home_prob(self) -> float:
        """Raw implied probability (with vig)."""
        return 1 / self.home_odds
    
    @computed_field
    @property
    def implied_away_prob(self) -> float:
        return 1 / self.away_odds
    
    @computed_field
    @property
    def implied_draw_prob(self) -> Optional[float]:
        if self.draw_odds:
            return 1 / self.draw_odds
        return None
    
    def no_vig_probs(self) -> Dict[str, float]:
        """Remove vig to get true probabilities."""
        total = self.implied_home_prob + self.implied_away_prob
        if self.draw_odds:
            total += self.implied_draw_prob
        
        result = {
            "home": self.implied_home_prob / total,
            "away": self.implied_away_prob / total,
        }
        if self.draw_odds:
            result["draw"] = self.implied_draw_prob / total
        return result


class BestOdds(BaseModel):
    """
    Best available odds across all bookmakers for a match.
    
    This is what we actually bet on - the best price in the market.
    """
    
    match_id: str
    timestamp: datetime
    
    # Best odds per outcome
    home_odds: float
    home_bookmaker: str
    draw_odds: Optional[float] = None
    draw_bookmaker: Optional[str] = None
    away_odds: float
    away_bookmaker: str
    
    # Number of bookmakers offering each outcome
    home_book_count: int = 1
    draw_book_count: int = 0
    away_book_count: int = 1
    
    # Outlier flags (for guardrails)
    home_is_outlier: bool = False
    draw_is_outlier: bool = False
    away_is_outlier: bool = False
    
    @computed_field
    @property
    def market_overround(self) -> float:
        """Combined market margin using best odds."""
        probs = 1/self.home_odds + 1/self.away_odds
        if self.draw_odds:
            probs += 1/self.draw_odds
        return (probs - 1) * 100
    
    def get_odds(self, outcome: Outcome) -> float:
        """Get best odds for given outcome."""
        if outcome == Outcome.HOME:
            return self.home_odds
        elif outcome == Outcome.DRAW:
            return self.draw_odds or 0
        else:
            return self.away_odds


class LineMovement(BaseModel):
    """
    Tracks odds movement over time for a single outcome.
    
    Key for detecting:
    - Sharp money (sudden moves)
    - Steam moves (coordinated drops)
    - Public bias (slow drift)
    """
    
    match_id: str
    outcome: Outcome
    
    # Opening and current
    opening_odds: float
    current_odds: float
    
    # Timestamps
    first_seen: datetime
    last_updated: datetime
    
    # Movement stats
    num_snapshots: int = 1
    max_odds: float = 0
    min_odds: float = 0
    
    @computed_field
    @property
    def total_movement_pct(self) -> float:
        """Total price change as percentage."""
        return ((self.current_odds / self.opening_odds) - 1) * 100
    
    @computed_field
    @property
    def is_steaming(self) -> bool:
        """Odds dropping significantly = sharp money."""
        return self.total_movement_pct < -10
    
    @computed_field
    @property
    def is_drifting(self) -> bool:
        """Odds rising significantly = public fade."""
        return self.total_movement_pct > 10


class MatchResult(BaseModel):
    """
    Final match result for settlement and backtesting.
    """
    
    match_id: str
    home_score: int
    away_score: int
    settled_at: datetime
    
    # For totals market
    @computed_field
    @property
    def total_goals(self) -> int:
        return self.home_score + self.away_score
    
    @computed_field
    @property
    def outcome(self) -> Outcome:
        if self.home_score > self.away_score:
            return Outcome.HOME
        elif self.home_score < self.away_score:
            return Outcome.AWAY
        return Outcome.DRAW
    
    @computed_field
    @property
    def btts(self) -> bool:
        """Both teams to score."""
        return self.home_score > 0 and self.away_score > 0
    
    @computed_field
    @property
    def over_2_5(self) -> bool:
        return self.total_goals > 2.5


class ClosingOdds(BaseModel):
    """
    Closing odds snapshot - captured just before match starts.
    
    CRITICAL for CLV tracking - the gold standard for measuring edge.
    If you consistently beat closing odds, you have a real edge.
    """
    
    match_id: str
    captured_at: datetime
    minutes_before_kickoff: int
    
    # Pinnacle closing (the sharpest line)
    pinnacle_home: Optional[float] = None
    pinnacle_draw: Optional[float] = None
    pinnacle_away: Optional[float] = None
    
    # Market average closing
    avg_home: float
    avg_draw: Optional[float] = None
    avg_away: float
    
    # Best closing
    best_home: float
class Player(BaseModel):
    """Player in a lineup."""
    id: str
    name: str
    position: Optional[str] = None
    rating: Optional[float] = None
    
class TeamLineup(BaseModel):
    """Lineup for one team."""
    team_name: str
    formation: Optional[str] = None
    starting_xi: List[Player] = []
    substitutes: List[Player] = []
    
class MatchLineups(BaseModel):
    """Lineups for both teams in a match."""
    match_id: str
    home: TeamLineup
    away: TeamLineup

class MatchStats(BaseModel):
    """
    Detailed match statistics (xG, shots, possession).
    
    Essential for 'AdvancedFeatureBuilder'.
    """
    match_id: str
    
    # Expected Goals
    xg_home: Optional[float] = None
    xg_away: Optional[float] = None
    
    # Possession
    possession_home: Optional[float] = None
    possession_away: Optional[float] = None
    
    # Shots
    shots_home: Optional[int] = None
    shots_away: Optional[int] = None
    shots_on_target_home: Optional[int] = None
    shots_on_target_away: Optional[int] = None
    
    
    # Corners
    corners_home: Optional[int] = None
    corners_away: Optional[int] = None
    
    # Fouls & Cards
    fouls_home: Optional[int] = None
    fouls_away: Optional[int] = None
    yellow_cards_home: Optional[int] = None
    yellow_cards_away: Optional[int] = None
    red_cards_home: Optional[int] = None
    red_cards_away: Optional[int] = None


class RefereeInfo(BaseModel):
    """Referee data for a match."""
    id: Optional[str] = None
    name: str
    
class WeatherInfo(BaseModel):
    """Weather data for a match venue."""
    temperature_c: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    humidity_pct: Optional[float] = None
    precipitation_mm: Optional[float] = None
    description: Optional[str] = None  # e.g. "Rain", "Clear"
    
class InjuryInfo(BaseModel):
    """Injury/suspension for a player."""
    player_name: str
    player_id: Optional[str] = None
    reason: str = "unknown"  # "injury", "suspension", "illness"
    expected_return: Optional[str] = None
    
class CoachInfo(BaseModel):
    """Coach information for a team."""
    name: str
    coach_id: Optional[str] = None
    appointed_date: Optional[str] = None  # ISO date string

class VenueInfo(BaseModel):
    """Venue information."""
    name: Optional[str] = None
    city: Optional[str] = None
    capacity: Optional[int] = None
    surface: Optional[str] = None  # "grass", "artificial"
    altitude_m: Optional[int] = None

class MatchEnrichment(BaseModel):
    """Additional enrichment data attached to a Match."""
    referee: Optional[RefereeInfo] = None
    weather: Optional[WeatherInfo] = None
    home_injuries: List[InjuryInfo] = []
    away_injuries: List[InjuryInfo] = []
    home_coach: Optional[CoachInfo] = None
    away_coach: Optional[CoachInfo] = None
    venue_info: Optional[VenueInfo] = None
    sm_odds_home: Optional[float] = None
    sm_odds_draw: Optional[float] = None
    sm_odds_away: Optional[float] = None
    
    # New Multi-Market Odds
    sm_corners_home: Optional[float] = None
    sm_corners_draw: Optional[float] = None
    sm_corners_away: Optional[float] = None
    sm_btts_yes: Optional[float] = None
    sm_btts_no: Optional[float] = None

# Update forward refs
Match.model_rebuild()

