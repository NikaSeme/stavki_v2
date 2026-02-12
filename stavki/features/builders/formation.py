"""
Formation / Tactical Feature Builder (Tier 2).

Extracts formation archetypes from lineup data:
- Defensive (5-x-x), Balanced (4-x-x), Attacking (3-x-x)
- Formation matchup scoring
- Tactical style indicators
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# Formation archetype classification
FORMATION_ARCHETYPES = {
    "defensive": {"5-4-1", "5-3-2", "5-2-3", "5-2-1-2", "4-5-1"},
    "balanced": {"4-4-2", "4-3-3", "4-2-3-1", "4-1-4-1", "4-4-1-1", "4-3-2-1"},
    "attacking": {"3-5-2", "3-4-3", "3-4-2-1", "3-3-4", "3-4-1-2"},
}


def _classify_formation(formation: Optional[str]) -> str:
    """Classify a formation string into an archetype."""
    if not formation:
        return "unknown"
    f = formation.strip()
    for archetype, patterns in FORMATION_ARCHETYPES.items():
        if f in patterns:
            return archetype
    # Fallback: check first digit
    try:
        defenders = int(f.split("-")[0])
        if defenders >= 5:
            return "defensive"
        elif defenders <= 3:
            return "attacking"
        return "balanced"
    except (ValueError, IndexError):
        return "unknown"


def _formation_score(formation: Optional[str]) -> float:
    """Score a formation on a defensive (0.0) to attacking (1.0) scale."""
    archetype = _classify_formation(formation)
    mapping = {"defensive": 0.2, "balanced": 0.5, "attacking": 0.8, "unknown": 0.5}
    return mapping[archetype]


class FormationFeatureBuilder:
    """
    Compute formation/tactical matchup features.
    
    Uses lineup.formation from the enrichment data.
    """
    
    name = "formation"
    
    def __init__(self):
        # Track team formation preferences
        self._team_formations: Dict[str, List[str]] = {}
        self._is_fitted = False
    
    def fit(self, matches: List[Match]) -> None:
        """Track formation preferences per team."""
        self._team_formations.clear()
        team_fmts: Dict[str, List[str]] = {}
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.lineups:
                continue
            
            home = m.home_team.normalized_name
            away = m.away_team.normalized_name
            
            if m.lineups.home.formation:
                if home not in team_fmts:
                    team_fmts[home] = []
                team_fmts[home].append(m.lineups.home.formation)
                team_fmts[home] = team_fmts[home][-10:]
            
            if m.lineups.away.formation:
                if away not in team_fmts:
                    team_fmts[away] = []
                team_fmts[away].append(m.lineups.away.formation)
                team_fmts[away] = team_fmts[away][-10:]
        
        self._team_formations = team_fmts
        self._is_fitted = True
        logger.info(f"FormationFeatureBuilder: {len(team_fmts)} teams profiled")
    
    def _get_preferred_style(self, team: str) -> float:
        """Get team's average formation score (defensive to attacking)."""
        fmts = self._team_formations.get(team, [])
        if not fmts:
            return 0.5
        scores = [_formation_score(f) for f in fmts]
        return sum(scores) / len(scores)
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get formation features for a match."""
        defaults = {
            "formation_score_home": 0.5,
            "formation_score_away": 0.5,
            "formation_mismatch": 0.0,
            "home_style_attacking": 0.5,
            "away_style_attacking": 0.5,
        }
        
        if not match:
            return defaults
        
        features = {}
        
        # Current match formations (if available from lineup)
        home_fmt = None
        away_fmt = None
        if match.lineups:
            home_fmt = match.lineups.home.formation
            away_fmt = match.lineups.away.formation
        
        home_score = _formation_score(home_fmt) if home_fmt else self._get_preferred_style(
            match.home_team.normalized_name
        )
        away_score = _formation_score(away_fmt) if away_fmt else self._get_preferred_style(
            match.away_team.normalized_name
        )
        
        features["formation_score_home"] = round(home_score, 2)
        features["formation_score_away"] = round(away_score, 2)
        features["formation_mismatch"] = round(abs(home_score - away_score), 2)
        features["home_style_attacking"] = round(
            self._get_preferred_style(match.home_team.normalized_name), 2
        )
        features["away_style_attacking"] = round(
            self._get_preferred_style(match.away_team.normalized_name), 2
        )
        
        return features
