"""
League Router and Blending Engine
=================================

Provides:
1. LeagueRouter - Per-league strategy and model weights from config
2. LiquidityBlender - Smart blending based on market efficiency

Key insight: Elite leagues (EPL, La Liga) have sharper markets,
so we trust market prices more. Lower leagues have inefficient markets,
so we trust our model more.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# Liquidity tier definitions
TIER_1_LEAGUES = {
    # Elite - Market is extremely sharp
    "soccer_epl", "EPL", "E0", "Premier League",
    "soccer_spain_la_liga", "LaLiga", "SP1",
    "soccer_uefa_champions_league", "UCL",
}

TIER_2_LEAGUES = {
    # Major domestic - Sharp but beatable
    "soccer_italy_serie_a", "SerieA", "I1",
    "soccer_germany_bundesliga", "Bundesliga", "D1",
    "soccer_france_ligue_one", "Ligue1", "F1",
    "soccer_efl_champ", "Championship", "E1",
    "soccer_netherlands_eredivisie", "Eredivisie",
}

# Team name normalization — single source of truth
from stavki.data.processors.normalize import (
    normalize_team_name,
    TEAM_ALIASES,
)


@dataclass
class LeagueConfig:
    """Configuration for a specific league."""
    name: str
    policy: str  # "BET", "SKIP", "CAUTIOUS"
    weights: Dict[str, float]  # model -> weight
    min_ev: float = 0.03
    kelly_fraction: float = 0.25
    tier: str = "tier3"


class LeagueRouter:
    """
    Decides betting strategy and model weights based on league.
    
    Loads configuration from JSON file, falls back to defaults.
    """
    
    DEFAULT_CONFIG = {
        "default": {
            "policy": "BET",
            "weights": {"catboost": 0.35, "neural": 0.35, "poisson": 0.30},
            "min_ev": 0.03,
            "kelly_fraction": 0.25,
        },
        "leagues": {}
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self._league_cache: Dict[str, LeagueConfig] = {}
    
    def _load_config(self) -> dict:
        """Load configuration from file or use defaults."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    config = json.load(f)
                logger.info(f"Loaded league config from {self.config_path}")
                return {**self.DEFAULT_CONFIG, **config}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return self.DEFAULT_CONFIG.copy()
    
    def get_config(self, league: str) -> LeagueConfig:
        """Get configuration for a league."""
        # Normalize league name
        league_key = league.lower().replace(" ", "_").replace("-", "_")
        
        # Check cache
        if league_key in self._league_cache:
            return self._league_cache[league_key]
        
        # Check config file
        if league_key in self.config.get("leagues", {}):
            league_conf = self.config["leagues"][league_key]
            config = LeagueConfig(
                name=league,
                policy=league_conf.get("policy", "BET"),
                weights=league_conf.get("weights", self.DEFAULT_CONFIG["default"]["weights"]),
                min_ev=league_conf.get("min_ev", 0.03),
                kelly_fraction=league_conf.get("kelly_fraction", 0.25),
                tier=self._get_tier(league),
            )
        else:
            # Use defaults
            config = LeagueConfig(
                name=league,
                policy=self.DEFAULT_CONFIG["default"]["policy"],
                weights=self.DEFAULT_CONFIG["default"]["weights"],
                min_ev=self.DEFAULT_CONFIG["default"]["min_ev"],
                kelly_fraction=self.DEFAULT_CONFIG["default"]["kelly_fraction"],
                tier=self._get_tier(league),
            )
        
        self._league_cache[league_key] = config
        return config
    
    def _get_tier(self, league: str) -> str:
        """Determine liquidity tier for a league."""
        league_check = league.lower().replace(" ", "_")
        
        if any(t.lower().replace(" ", "_") in league_check or league_check in t.lower() 
               for t in TIER_1_LEAGUES):
            return "tier1"
        elif any(t.lower().replace(" ", "_") in league_check or league_check in t.lower() 
                 for t in TIER_2_LEAGUES):
            return "tier2"
        return "tier3"
    
    def get_weights(self, league: str) -> Tuple[float, float, float]:
        """Get model weights tuple (catboost, neural, poisson)."""
        config = self.get_config(league)
        w = config.weights
        return (
            w.get("catboost", 0.35),
            w.get("neural", 0.35),
            w.get("poisson", 0.30),
        )
    
    def should_bet(self, league: str) -> bool:
        """Check if league policy allows betting."""
        config = self.get_config(league)
        return config.policy != "SKIP"
    
    def save_config(self, filepath: Path):
        """Save current configuration to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to {filepath}")


class LiquidityBlender:
    """
    Smart blending between model and market probabilities.
    
    Logic:
    - High Liquidity (Tier 1): Trust market more (low alpha)
    - Low Liquidity (Tier 3): Trust model more (high alpha)
    
    alpha = weight given to model probability
    final_prob = alpha * p_model + (1 - alpha) * p_market
    """
    
    # Default alpha values per tier
    # Default alpha values per tier
    # Reduced from 1.0 (Bob AI-Only) to mathematically sound liquidity distributions
    # Tier 1 defaults to 0.40 since the Bookmaker market is exceptionally sharp
    DEFAULT_ALPHAS = {
        "tier1": 0.40,  # 40% model, 60% market (Elite Leagues)
        "tier2": 0.60,  # 60% model, 40% market (Major Domestic)
        "tier3": 0.85,  # 85% model, 15% market (Lower Leagues)
    }
    
    def __init__(
        self,
        alphas: Optional[Dict[str, float]] = None,
        league_router: Optional[LeagueRouter] = None,
    ):
        self.alphas = {**self.DEFAULT_ALPHAS, **(alphas or {})}
        self.router = league_router or LeagueRouter()
    
    def blend(
        self,
        p_model: float,
        p_market: float,
        league: str = "unknown",
        alpha_override: Optional[float] = None,
    ) -> float:
        """
        Blend model and market probabilities.
        
        Args:
            p_model: Model probability
            p_market: Market (no-vig) probability
            league: League for tier lookup
            alpha_override: Force specific alpha
        
        Returns:
            Blended probability
        """
        if alpha_override is not None:
            alpha = alpha_override
        else:
            tier = self.router._get_tier(league)
            alpha = self.alphas.get(tier, 0.50)
        
        return alpha * p_model + (1 - alpha) * p_market
    
    def get_alpha(self, league: str) -> float:
        """Get alpha (model weight) for a league."""
        tier = self.router._get_tier(league)
        return self.alphas.get(tier, 0.50)
    
    def optimize_alphas(
        self,
        historical_bets: List[Dict],
        leagues: List[str],
    ) -> Dict[str, float]:
        """
        Optimize alpha values per tier/league.
        
        Uses grid search to find best alpha for each tier.
        """
        tier_bets = {"tier1": [], "tier2": [], "tier3": []}
        
        for bet in historical_bets:
            league = bet.get("league", "unknown")
            tier = self.router._get_tier(league)
            tier_bets[tier].append(bet)
        
        optimized = {}
        
        for tier, bets in tier_bets.items():
            if len(bets) < 50:
                optimized[tier] = self.DEFAULT_ALPHAS[tier]
                continue
            
            best_alpha = 0.50
            best_score = float("-inf")
            
            for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                # Simulate with this alpha
                total_staked = 0
                total_profit = 0
                
                for bet in bets:
                    p_model = bet.get("p_model")
                    p_market = bet.get("p_market")
                    odds = bet.get("odds")
                    result = bet.get("result")
                    
                    if not all([p_model, p_market, odds, result]):
                        continue
                    
                    p_blend = alpha * p_model + (1 - alpha) * p_market
                    ev = p_blend * odds - 1
                    
                    if ev > 0.03:  # EV threshold
                        stake = 10
                        total_staked += stake
                        
                        if result == "win":
                            total_profit += stake * (odds - 1)
                        else:
                            total_profit -= stake
                
                roi = total_profit / total_staked if total_staked > 0 else 0
                
                if roi > best_score:
                    best_score = roi
                    best_alpha = alpha
            
            optimized[tier] = best_alpha
            logger.info(f"{tier}: optimal alpha = {best_alpha:.2f}")
        
        self.alphas = optimized
        return optimized


def check_model_market_divergence(
    p_model: float,
    p_market: float,
    max_divergence: float = 0.20,
) -> Tuple[bool, float, str]:
    """
    Check if model probability diverges too much from market.
    
    Large divergence might indicate model error or special information.
    
    Returns:
        (is_safe, divergence, level)
    """
    divergence = abs(p_model - p_market)
    
    if divergence <= max_divergence:
        level = "safe"
        is_safe = True
    elif divergence <= max_divergence * 2:
        level = "caution"
        is_safe = False
    else:
        level = "extreme"
        is_safe = False
    
    return is_safe, divergence, level


def check_outlier_odds(
    all_odds: List[float],
    gap_threshold: float = 0.20,
) -> bool:
    """
    Check if best odds is an outlier (>20% above second-best).
    
    Outlier odds often indicate bookmaker error or trap.
    """
    if len(all_odds) < 2:
        return False
    
    sorted_odds = sorted(all_odds, reverse=True)
    best = sorted_odds[0]
    second = sorted_odds[1]
    
    gap = (best - second) / second
    return gap > gap_threshold


def calculate_justified_score(
    p_model: float,
    p_market: float,
    odds: float,
    ev: float,
) -> int:
    """
    Calculate 0-100 score for how justified a value bet is.
    
    High scores = likely real value
    Low scores = likely model error
    """
    score = 100
    
    # Penalty for model-market divergence
    divergence = abs(p_model - p_market)
    if divergence > 0.40:
        score -= 60
    elif divergence > 0.20:
        score -= 40
    elif divergence > 0.10:
        score -= 20
    
    # Penalty for very high odds
    if odds > 20:
        score -= 30
    elif odds > 10:
        score -= 20
    elif odds > 6:
        score -= 10
    
    # Penalty for extreme EV
    if ev > 2.0 and divergence > 0.30:
        score -= 40
    elif ev > 1.0 and divergence > 0.20:
        score -= 20
    
    return max(0, min(100, score))
