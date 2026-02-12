"""
Data processors package.

Exports normalization, validation, and line movement processors.
"""

from .normalize import (
    normalize_team_name,
    add_team_alias,
    suggest_match,
    SourceNormalizer,
    TEAM_ALIASES,
)

from .validate import (
    ValidationResult,
    OddsValidator,
    MatchValidator,
    DataQualityMonitor,
)

from .line_movement import (
    LineMovementTracker,
    SharpMoneyDetector,
    CLVTracker,
)


__all__ = [
    # Normalize
    "normalize_team_name",
    "add_team_alias",
    "suggest_match",
    "SourceNormalizer",
    "TEAM_ALIASES",
    # Validate
    "ValidationResult",
    "OddsValidator",
    "MatchValidator",
    "DataQualityMonitor",
    # Line movement
    "LineMovementTracker",
    "SharpMoneyDetector",
    "CLVTracker",
]
