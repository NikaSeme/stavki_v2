"""
Data package - collection, storage, and processing.

This is the entry point for all data operations in STAVKI.
"""

from .schemas import (
    League, Outcome, Team, Match, OddsSnapshot, BestOdds,
    LineMovement, MatchResult, ClosingOdds,
    BetStatus, Prediction, ValueSignal, BetRecommendation,
    PlacedBet, DailyStats,
    TeamFeatures, H2HFeatures, MatchFeatures, LeagueStats,
)

from .collectors import (
    OddsAPIClient, OddsAPICollector, 
    FootballDataLoader, HistoricalOddsExtractor,
)

from .processors import (
    normalize_team_name, OddsValidator, MatchValidator,
    LineMovementTracker, CLVTracker,
)

from .storage import Database


__all__ = [
    # Schemas
    "League", "Outcome", "Team", "Match", "OddsSnapshot", "BestOdds",
    "LineMovement", "MatchResult", "ClosingOdds",
    "BetStatus", "Prediction", "ValueSignal", "BetRecommendation",
    "PlacedBet", "DailyStats",
    "TeamFeatures", "H2HFeatures", "MatchFeatures", "LeagueStats",
    # Collectors
    "OddsAPIClient", "OddsAPICollector",
    "FootballDataLoader", "HistoricalOddsExtractor",
    # Processors
    "normalize_team_name", "OddsValidator", "MatchValidator",
    "LineMovementTracker", "CLVTracker",
    # Storage
    "Database",
]
