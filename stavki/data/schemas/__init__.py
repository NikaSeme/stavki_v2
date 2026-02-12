"""
Data schemas package.

Re-exports all schema classes for convenient importing:
    from stavki.data.schemas import Match, OddsSnapshot, Prediction
"""

from .match import (
    League,
    Outcome,
    Team,
    Match,
    OddsSnapshot,
    BestOdds,
    LineMovement,
    MatchResult,
    ClosingOdds,
    MatchStats,
    MatchLineups,
    TeamLineup,
    Player,
    MatchEnrichment,
    RefereeInfo,
    WeatherInfo,
    InjuryInfo,
    CoachInfo,
    VenueInfo,
)

from .prediction import (
    BetStatus,
    Prediction,
    ValueSignal,
    BetRecommendation,
    PlacedBet,
    DailyStats,
)

from .features import (
    TeamFeatures,
    H2HFeatures,
    MatchFeatures,
    LeagueStats,
)


__all__ = [
    # Match schemas
    "League",
    "Outcome",
    "Team",
    "Match",
    "OddsSnapshot",
    "BestOdds",
    "LineMovement",
    "MatchResult",
    "ClosingOdds",
    "MatchStats",
    "MatchLineups",
    "TeamLineup",
    "Player",
    "MatchEnrichment",
    "RefereeInfo",
    "WeatherInfo",
    "InjuryInfo",
    "CoachInfo",
    "VenueInfo",
    # Prediction schemas
    "BetStatus",
    "Prediction",
    "ValueSignal",
    "BetRecommendation",
    "PlacedBet",
    "DailyStats",
    # Feature schemas
    "TeamFeatures",
    "H2HFeatures",
    "MatchFeatures",
    "LeagueStats",
]
