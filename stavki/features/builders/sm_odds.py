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
            
        features = defaults.copy()
        
        if match.enrichment.sm_odds_home and match.enrichment.sm_odds_draw and match.enrichment.sm_odds_away:
            sm_home = match.enrichment.sm_odds_home
            sm_draw = match.enrichment.sm_odds_draw
            sm_away = match.enrichment.sm_odds_away
            
            # Convert odds to implied probabilities (with vig removal)
            raw_sum = (1/sm_home) + (1/sm_draw) + (1/sm_away)
            features.update({
                "sm_implied_home": round((1/sm_home) / raw_sum, 4),
                "sm_implied_draw": round((1/sm_draw) / raw_sum, 4),
                "sm_implied_away": round((1/sm_away) / raw_sum, 4),
            })
            
            # Disagreement with primary source
            if primary_home_prob and primary_draw_prob and primary_away_prob:
                disagreement = (
                    abs(features["sm_implied_home"] - primary_home_prob) +
                    abs(features["sm_implied_draw"] - primary_draw_prob) +
                    abs(features["sm_implied_away"] - primary_away_prob)
                ) / 3
                features["odds_source_disagreement"] = round(disagreement, 4)

        # Corners 1X2 Features
        if match.enrichment.sm_corners_home and match.enrichment.sm_corners_away:
             c_home = match.enrichment.sm_corners_home
             c_draw = match.enrichment.sm_corners_draw or 100.0 # fallback if missing
             c_away = match.enrichment.sm_corners_away
             
             raw_sum_c = (1/c_home) + (1/c_draw) + (1/c_away)
             features.update({
                 "sm_corners_implied_home": round((1/c_home) / raw_sum_c, 4),
                 "sm_corners_implied_away": round((1/c_away) / raw_sum_c, 4),
             })

        # BTTS Features
        if match.enrichment.sm_btts_yes and match.enrichment.sm_btts_no:
            b_yes = match.enrichment.sm_btts_yes
            b_no = match.enrichment.sm_btts_no
            
            raw_sum_b = (1/b_yes) + (1/b_no)
            features.update({
                "sm_btts_implied_yes": round((1/b_yes) / raw_sum_b, 4),
                "sm_btts_implied_no": round((1/b_no) / raw_sum_b, 4),
            })
        
        return features
