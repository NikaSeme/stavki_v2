"""
SportMonks Odds Feature Builder (Tier 3).

Cross-references SportMonks pre-match odds with existing odds data:
- Implied probabilities from SM odds
- Disagreement with primary odds source
- Market consensus signal
"""

from typing import Dict, Optional
from datetime import datetime
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)


class SMOddsFeatureBuilder:
    """
    Compute SportMonks odds cross-reference features.
    
    Uses enrichment SM odds to detect disagreement between
    odds sources — a powerful signal for value bets.
    """
    
    name = "sm_odds"
    
    def get_features(
        self,
        match: Optional[Match] = None,
        primary_home_prob: Optional[float] = None,
        primary_draw_prob: Optional[float] = None,
        primary_away_prob: Optional[float] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Get SM odds features.
        
        Args:
            match: Match with enrichment.sm_odds
            primary_*_prob: Implied probs from primary odds source (Odds API / CSV)
        """
        defaults = {
            "sm_implied_home": 0.33,
            "sm_implied_draw": 0.33,
            "sm_implied_away": 0.34,
            "odds_source_disagreement": 0.0,
        }
        
        if not match or not match.enrichment:
            return defaults
        
        sm_home = match.enrichment.sm_odds_home
        sm_draw = match.enrichment.sm_odds_draw
        sm_away = match.enrichment.sm_odds_away
        
        if not all([sm_home, sm_draw, sm_away]):
            return defaults
        
        # Convert odds to implied probabilities (with vig removal)
        raw_sum = (1/sm_home) + (1/sm_draw) + (1/sm_away)
        sm_p_home = (1/sm_home) / raw_sum
        sm_p_draw = (1/sm_draw) / raw_sum
        sm_p_away = (1/sm_away) / raw_sum
        
        features = {
            "sm_implied_home": round(sm_p_home, 4),
            "sm_implied_draw": round(sm_p_draw, 4),
            "sm_implied_away": round(sm_p_away, 4),
        }
        
        # Disagreement with primary source
        if primary_home_prob and primary_draw_prob and primary_away_prob:
            # Mean absolute difference between probability distributions
            disagreement = (
                abs(sm_p_home - primary_home_prob) +
                abs(sm_p_draw - primary_draw_prob) +
                abs(sm_p_away - primary_away_prob)
            ) / 3
            features["odds_source_disagreement"] = round(disagreement, 4)
        else:
            features["odds_source_disagreement"] = 0.0
        
        return features
