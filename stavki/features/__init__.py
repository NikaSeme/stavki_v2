"""
Features package - feature engineering for betting models.

Usage:
    from stavki.features import FeatureRegistry
    
    registry = FeatureRegistry()
    registry.fit(historical_matches)
    features = registry.compute("manchester united", "chelsea")
"""

from .base import FeatureBuilder, TeamFeatureBuilder, RollingFeatureBuilder
from .registry import FeatureRegistry, build_features
from .builders.elo import EloCalculator, EloBuilder
from .builders.form import FormCalculator, FormBuilder, GoalsBuilder
from .builders.h2h import H2HBuilder
from .builders.disagreement import (
    DisagreementBuilder,
    calculate_disagreement,
    detect_contrarian_opportunity,
    calculate_confidence_score,
)
from .builders.advanced_stats import AdvancedFeatureBuilder
from .builders.roster import RosterFeatureBuilder


__all__ = [
    # Base classes
    "FeatureBuilder",
    "TeamFeatureBuilder", 
    "RollingFeatureBuilder",
    # Registry
    "FeatureRegistry",
    "build_features",
    # ELO
    "EloCalculator",
    "EloBuilder",
    # Form
    "FormCalculator",
    "FormBuilder",
    "GoalsBuilder",
    # H2H
    "H2HBuilder",
    # Disagreement
    "DisagreementBuilder",
    "calculate_disagreement",
    "detect_contrarian_opportunity",
    "calculate_confidence_score",
    # Advanced
    "AdvancedFeatureBuilder",
    "RosterFeatureBuilder",
]
