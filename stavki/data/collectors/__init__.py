"""
Data collectors package.

Exports API clients and data loaders.
"""

from .odds_api import (
    OddsAPIClient,
    OddsAPICollector,
    APIResponse,
)

from .historical import (
    FootballDataLoader,
    HistoricalOddsExtractor,
    FOOTBALL_DATA_LEAGUES,
)

from .closing_odds import ClosingOddsCollector


__all__ = [
    # Odds API
    "OddsAPIClient",
    "OddsAPICollector",
    "APIResponse",
    # Historical
    "FootballDataLoader",
    "HistoricalOddsExtractor",
    "FOOTBALL_DATA_LEAGUES",
    # CLV Tracking
    "ClosingOddsCollector",
]
