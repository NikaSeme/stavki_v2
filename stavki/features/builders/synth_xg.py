"""
Synthetic xG Feature Builder (Tier 1).

Estimates expected goals from per-player shot data when official xG
is not available on the API plan.

Uses calibrated coefficients:
  synth_xG = 0.03 * shots + 0.12 * shots_on_target + 0.35 * big_chances + 0.05

Features produced:
  - synth_xg_home / synth_xg_away — per-match estimated xG
  - synth_xg_diff — home advantage differential
  - synth_xg_overperform_home / _away — actual goals vs expected (luck signal)
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import logging
import json

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# StatsBomb-derived coefficients (academic baseline)
DEFAULT_COEFS = {
    "shots": 0.03,
    "sot": 0.12,
    "big_chances": 0.35,
    "intercept": 0.05,
}


class SyntheticXGBuilder:
    """
    Compute synthetic xG features from per-player shot data.
    
    Uses historical match data to build rolling team xG averages,
    then predicts expected goals for upcoming matches.
    """
    
    name = "synth_xg"
    
    def __init__(self, rolling_window: int = 10):
        self.rolling_window = rolling_window
        self.coefs = DEFAULT_COEFS.copy()
        # team -> list of recent { synth_xg, actual_goals }
        self._team_xg_history: Dict[str, list] = defaultdict(list)
        self._is_fitted = False
    
    def _compute_match_xg(self, shots: float, sot: float,
                          big_chances: float) -> float:
        """Compute synthetic xG from shot data."""
        xg = (self.coefs["shots"] * shots +
              self.coefs["sot"] * sot +
              self.coefs["big_chances"] * big_chances +
              self.coefs["intercept"])
        return max(0.0, round(xg, 3))
    
    def _extract_team_shots(self, match: Match, side: str) -> dict:
        """Extract shot data from match lineups or stats."""
        result = {"shots": 0, "sot": 0, "big_chances": 0}
        
        if not match.lineups:
            # Fall back to match stats
            if match.stats:
                if side == "home":
                    result["shots"] = match.stats.shots_home or 0
                    result["sot"] = match.stats.shots_on_target_home or 0
                else:
                    result["shots"] = match.stats.shots_away or 0
                    result["sot"] = match.stats.shots_on_target_away or 0
            return result
        
        # Get per-player data from lineups
        lineup = match.lineups.home if side == "home" else match.lineups.away
        if not lineup or not lineup.starting_xi:
            return result
        
        for p in lineup.starting_xi:
            # PlayerEntry might have extra fields if populated from enriched data
            player_dict = p.model_dump() if hasattr(p, 'model_dump') else {}
            result["shots"] += player_dict.get("shots", 0) or 0
            result["sot"] += player_dict.get("shots_on_target", 0) or 0
            bc = (player_dict.get("big_chances_created", 0) or 0) + \
                 (player_dict.get("big_chances_missed", 0) or 0)
            result["big_chances"] += bc
        
        # If no per-player shot data, fall back to match stats
        if result["shots"] == 0 and match.stats:
            if side == "home":
                result["shots"] = match.stats.shots_home or 0
                result["sot"] = match.stats.shots_on_target_home or 0
            else:
                result["shots"] = match.stats.shots_away or 0
                result["sot"] = match.stats.shots_on_target_away or 0
        
        return result
    
    def fit(self, matches: List[Match]) -> None:
        """Build rolling xG history per team from historical matches."""
        self._team_xg_history.clear()
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            for side, team_name, goals in [
                ("home", m.home_team.normalized_name, m.home_score),
                ("away", m.away_team.normalized_name, m.away_score),
            ]:
                shot_data = self._extract_team_shots(m, side)
                synth_xg = self._compute_match_xg(
                    shot_data["shots"], shot_data["sot"],
                    shot_data["big_chances"]
                )
                
                self._team_xg_history[team_name].append({
                    "xg": synth_xg,
                    "goals": goals or 0,
                    "date": m.commence_time,
                })
                
                # Trim to window
                if len(self._team_xg_history[team_name]) > self.rolling_window * 2:
                    self._team_xg_history[team_name] = \
                        self._team_xg_history[team_name][-self.rolling_window * 2:]
        
        self._is_fitted = True
        logger.info(f"SyntheticXGBuilder: {len(self._team_xg_history)} teams profiled")
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get synthetic xG features for a match."""
        defaults = {
            "synth_xg_home": 1.3,
            "synth_xg_away": 1.1,
            "synth_xg_diff": 0.2,
            "synth_xg_overperform_home": 0.0,
            "synth_xg_overperform_away": 0.0,
        }
        
        if not match:
            return defaults
        
        features = {}
        ref_time = as_of or match.commence_time
        
        for side, team_name in [
            ("home", match.home_team.normalized_name),
            ("away", match.away_team.normalized_name),
        ]:
            history = self._team_xg_history.get(team_name, [])
            if ref_time:
                history = [h for h in history if h["date"] < ref_time]
            recent = history[-self.rolling_window:]
            
            if recent:
                avg_xg = sum(h["xg"] for h in recent) / len(recent)
                avg_goals = sum(h["goals"] for h in recent) / len(recent)
                overperform = round(avg_goals - avg_xg, 3)
            else:
                avg_xg = defaults[f"synth_xg_{side}"]
                overperform = 0.0
            
            features[f"synth_xg_{side}"] = round(avg_xg, 3)
            features[f"synth_xg_overperform_{side}"] = overperform
        
        features["synth_xg_diff"] = round(
            features["synth_xg_home"] - features["synth_xg_away"], 3
        )
        
        return features
