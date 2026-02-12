"""
Referee Feature Builder (Tier 1).

Computes per-referee tendencies from historical match data:
- Cards per game (yellow + red)
- Fouls per game
- Home bias (home cards vs away cards ratio)
- Strictness z-score vs league average
- Goals per game (new)
- Over 2.5 goals rate (new)
- Home win percentage (new)
- Experience / total matches (new, log-scaled)
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import logging
import numpy as np

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)


class RefereeFeatureBuilder:
    """
    Compute referee tendency features.
    
    Uses historical match data to build a profile per referee,
    then returns features for a given match's assigned referee.
    """
    
    name = "referee"
    
    def __init__(self, min_matches: int = 5, window: int = 30):
        self.min_matches = min_matches
        self.window = window  # Consider last N matches per referee
        self._referee_history: Dict[str, list] = defaultdict(list)
        self._is_fitted = False
    
    def fit(self, matches: List[Match]) -> None:
        """Build referee profiles from historical matches."""
        self._referee_history.clear()
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            ref_name = None
            if m.enrichment and m.enrichment.referee:
                ref_name = m.enrichment.referee.name.lower().strip()
            
            if not ref_name:
                continue
            
            # Build record even without full stats (goals always available)
            home_score = m.home_score or 0
            away_score = m.away_score or 0
            total_goals = home_score + away_score
            
            record = {
                "date": m.commence_time,
                "goals": total_goals,
                "home_win": 1 if home_score > away_score else 0,
                "draw": 1 if home_score == away_score else 0,
                "over25": 1 if total_goals > 2 else 0,
                "yellow_home": 0,
                "yellow_away": 0,
                "red_home": 0,
                "red_away": 0,
                "fouls_home": 0,
                "fouls_away": 0,
            }
            
            if m.stats:
                record["yellow_home"] = m.stats.yellow_cards_home or 0
                record["yellow_away"] = m.stats.yellow_cards_away or 0
                record["red_home"] = m.stats.red_cards_home or 0
                record["red_away"] = m.stats.red_cards_away or 0
                record["fouls_home"] = m.stats.fouls_home or 0
                record["fouls_away"] = m.stats.fouls_away or 0
            
            self._referee_history[ref_name].append(record)
        
        self._is_fitted = True
        logger.info(f"RefereeFeatureBuilder: {len(self._referee_history)} referees profiled")
    
    def get_features(
        self,
        match: Optional[Match] = None,
        referee_name: Optional[str] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Get referee tendency features for a match.
        
        Returns defaults (league averages) if referee not found.
        """
        defaults = {
            "ref_cards_per_game": 3.5,    # Typical league avg
            "ref_fouls_per_game": 22.0,   # Typical league avg
            "ref_home_card_ratio": 0.5,   # Balanced
            "ref_strictness": 0.0,        # Neutral
            "ref_goals_per_game": 2.7,    # League avg
            "ref_over25_rate": 0.55,      # League avg
            "ref_home_win_rate": 0.45,    # League avg
            "ref_experience": 0.0,        # Unknown
        }
        
        # Get referee name
        ref = referee_name
        if not ref and match and match.enrichment and match.enrichment.referee:
            ref = match.enrichment.referee.name.lower().strip()
        
        if not ref or ref not in self._referee_history:
            return defaults
        
        # Get recent matches for this referee
        history = self._referee_history[ref]
        if as_of:
            history = [h for h in history if h["date"] < as_of]
        recent = history[-self.window:]
        
        if len(recent) < self.min_matches:
            return defaults
        
        n = len(recent)
        total_yellows = sum(h["yellow_home"] + h["yellow_away"] for h in recent)
        total_reds = sum(h["red_home"] + h["red_away"] for h in recent)
        total_fouls = sum(h["fouls_home"] + h["fouls_away"] for h in recent)
        home_cards = sum(h["yellow_home"] + h["red_home"] for h in recent)
        all_cards = total_yellows + total_reds
        
        cards_pg = all_cards / n
        fouls_pg = total_fouls / n
        home_ratio = home_cards / max(all_cards, 1)
        
        # Strictness: z-score vs defaults
        strictness = (cards_pg - defaults["ref_cards_per_game"]) / 1.5  # ~1.5 std
        
        # Goals per game
        total_goals = sum(h["goals"] for h in recent)
        goals_pg = total_goals / n
        
        # Over 2.5 rate
        over25_count = sum(h["over25"] for h in recent)
        over25_rate = over25_count / n
        
        # Home win rate
        home_wins = sum(h["home_win"] for h in recent)
        home_win_rate = home_wins / n
        
        # Experience (log-scaled total matches)
        total_matches = len(history)
        experience = round(np.log1p(total_matches), 2)
        
        return {
            "ref_cards_per_game": round(cards_pg, 2),
            "ref_fouls_per_game": round(fouls_pg, 2),
            "ref_home_card_ratio": round(home_ratio, 3),
            "ref_strictness": round(strictness, 3),
            "ref_goals_per_game": round(goals_pg, 2),
            "ref_over25_rate": round(over25_rate, 3),
            "ref_home_win_rate": round(home_win_rate, 3),
            "ref_experience": experience,
        }
