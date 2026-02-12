"""STAVKI Strategy Module - EV, Kelly, and Filters with Data-Driven Optimization."""

from .ev import EVCalculator, EVResult, compute_ev, filter_positive_ev
from .kelly import KellyStaker, StakeResult, kelly_simple
from .filters import BetFilters, MetaFilter, FilterResult
from .optimizer import WeightOptimizer, KellyOptimizer, ThresholdOptimizer
from .league_router import (
    LeagueRouter, LiquidityBlender, LeagueConfig,
    normalize_team_name, check_model_market_divergence,
    check_outlier_odds, calculate_justified_score,
    TEAM_ALIASES, TIER_1_LEAGUES, TIER_2_LEAGUES,
)

__all__ = [
    # EV
    "EVCalculator",
    "EVResult",
    "compute_ev",
    "filter_positive_ev",
    # Kelly
    "KellyStaker",
    "StakeResult",
    "kelly_simple",
    # Filters
    "BetFilters",
    "MetaFilter",
    "FilterResult",
    # Optimizer
    "WeightOptimizer",
    "KellyOptimizer",
    "ThresholdOptimizer",
    # League Router
    "LeagueRouter",
    "LiquidityBlender",
    "LeagueConfig",
    "normalize_team_name",
    "check_model_market_divergence",
    "check_outlier_odds",
    "calculate_justified_score",
    "TEAM_ALIASES",
    "TIER_1_LEAGUES",
    "TIER_2_LEAGUES",
]
