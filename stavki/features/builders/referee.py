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
        # Global averages computed from actual data during fit()
        self._global_avg: Dict[str, float] = {}
    
    def fit(self, matches: List[Match]) -> None:
        """Build referee profiles from historical matches."""
        self._referee_history.clear()
        
        # Collect ALL records first for global average computation
        all_records = []
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            ref_name = None
            if m.enrichment and m.enrichment.referee:
                ref_name = m.enrichment.referee.name.lower().strip()
            
            if not ref_name:
                continue
            
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
            all_records.append(record)
        
        # Compute global averages from ACTUAL fitted data
        if all_records:
            n = len(all_records)
            total_cards = sum(
                r["yellow_home"] + r["yellow_away"] + r["red_home"] + r["red_away"]
                for r in all_records
            )
            total_fouls = sum(r["fouls_home"] + r["fouls_away"] for r in all_records)
            total_goals = sum(r["goals"] for r in all_records)
            
            avg_cards = total_cards / n
            avg_fouls = total_fouls / n
            avg_goals = total_goals / n
            over25_rate = sum(r["over25"] for r in all_records) / n
            home_win_rate = sum(r["home_win"] for r in all_records) / n
            
            # Compute actual std of cards/game across referees (for strictness z-score)
            ref_cards_rates = []
            for ref_records in self._referee_history.values():
                if len(ref_records) >= self.min_matches:
                    ref_n = len(ref_records)
                    ref_total = sum(
                        r["yellow_home"] + r["yellow_away"] + r["red_home"] + r["red_away"]
                        for r in ref_records
                    )
                    ref_cards_rates.append(ref_total / ref_n)
            
            cards_std = float(np.std(ref_cards_rates)) if len(ref_cards_rates) > 1 else 1.0
            
            self._global_avg = {
                "cards_per_game": round(avg_cards, 3),
                "fouls_per_game": round(avg_fouls, 3),
                "goals_per_game": round(avg_goals, 3),
                "over25_rate": round(over25_rate, 3),
                "home_win_rate": round(home_win_rate, 3),
                "cards_std": round(max(cards_std, 0.1), 3),  # Floor at 0.1 to avoid div by 0
            }
            logger.info(
                f"RefereeFeatureBuilder: {len(self._referee_history)} referees profiled | "
                f"global avg: {avg_cards:.2f} cards/g, {avg_fouls:.1f} fouls/g, "
                f"{avg_goals:.2f} goals/g, {over25_rate:.1%} O2.5, {home_win_rate:.1%} home win"
            )
        else:
            self._global_avg = {}
            logger.info("RefereeFeatureBuilder: 0 referees profiled (no referee data)")
        
        self._is_fitted = True
    
    def _get_defaults(self) -> Dict[str, float]:
        """Return defaults from computed global averages, not hardcoded values."""
        ga = self._global_avg
        return {
            "ref_cards_per_game": ga.get("cards_per_game", 0.0),
            "ref_fouls_per_game": ga.get("fouls_per_game", 0.0),
            "ref_home_card_ratio": 0.5,   # Balanced is a valid default
            "ref_strictness": 0.0,        # Neutral z-score
            "ref_goals_per_game": ga.get("goals_per_game", 0.0),
            "ref_over25_rate": ga.get("over25_rate", 0.0),
            "ref_home_win_rate": ga.get("home_win_rate", 0.0),
            "ref_experience": 0.0,
        }
    
    def get_features(
        self,
        match: Optional[Match] = None,
        referee_name: Optional[str] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Get referee tendency features for a match.
        
        Returns dataset-computed averages if referee not found.
        """
        defaults = self._get_defaults()
        
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
        
        # Strictness: z-score vs ACTUAL global average (not hardcoded)
        global_cards_avg = self._global_avg.get("cards_per_game", cards_pg)
        cards_std = self._global_avg.get("cards_std", 1.0)
        strictness = (cards_pg - global_cards_avg) / cards_std
        
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
            "ref_total_cards_per_game": round((total_yellows + total_reds) / n, 2),
            "ref_fouls_per_game": round(fouls_pg, 2),
            "ref_home_card_ratio": round(home_ratio, 3),
            "ref_strictness": round(strictness, 3),
            "ref_goals_per_game": round(goals_pg, 2),
            "ref_over25_rate": round(over25_rate, 3),
            "ref_home_win_rate": round(home_win_rate, 3),
            "ref_experience": experience,
        }

