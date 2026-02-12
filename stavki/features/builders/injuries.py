"""
Injury / Suspension Feature Builder (Tier 1).

Assesses squad availability impact:
- Count of injured/suspended players
- Weighted impact (key players weighted more)
- Historical: inferred from lineup absences
- Live: from team injury reports
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)


class InjuryFeatureBuilder:
    """
    Compute injury/suspension impact features.
    
    For live matches: uses enrichment.home_injuries / away_injuries
    For historical: inferred from lineup regularity (missing regulars)
    """
    
    name = "injuries"
    
    def __init__(self):
        # Track recent starters per team
        self._recent_starters: Dict[str, set] = {}
        self._is_fitted = False
    
    def fit(self, matches: List[Match]) -> None:
        """Build recent starter profiles per team."""
        self._recent_starters.clear()
        
        # Track last 5 matches per team
        team_lineups: Dict[str, List[set]] = {}
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.lineups:
                continue
            
            home = m.home_team.normalized_name
            away = m.away_team.normalized_name
            
            home_xi = {p.id for p in m.lineups.home.starting_xi}
            away_xi = {p.id for p in m.lineups.away.starting_xi}
            
            if home not in team_lineups:
                team_lineups[home] = []
            team_lineups[home].append(home_xi)
            team_lineups[home] = team_lineups[home][-5:]
            
            if away not in team_lineups:
                team_lineups[away] = []
            team_lineups[away].append(away_xi)
            team_lineups[away] = team_lineups[away][-5:]
        
        # Regulars = appeared in >= 3 of last 5 lineups
        for team, lineups in team_lineups.items():
            if len(lineups) >= 3:
                all_players = set()
                for xi in lineups:
                    all_players.update(xi)
                regulars = set()
                for p in all_players:
                    count = sum(1 for xi in lineups if p in xi)
                    if count >= 3:
                        regulars.add(p)
                self._recent_starters[team] = regulars
        
        self._is_fitted = True
        logger.info(f"InjuryFeatureBuilder: {len(self._recent_starters)} teams profiled")
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get injury impact features."""
        defaults = {
            "injuries_count_home": 0,
            "injuries_count_away": 0,
            "regulars_missing_home": 0,
            "regulars_missing_away": 0,
        }
        
        if not match:
            return defaults
        
        features = {}
        
        # Direct injury data (from enrichment / live API)
        if match.enrichment:
            features["injuries_count_home"] = len(match.enrichment.home_injuries)
            features["injuries_count_away"] = len(match.enrichment.away_injuries)
        else:
            features["injuries_count_home"] = 0
            features["injuries_count_away"] = 0
        
        # Inferred absences (if we have lineup data)
        home_team = match.home_team.normalized_name
        away_team = match.away_team.normalized_name
        
        if match.lineups and self._is_fitted:
            home_xi = {p.id for p in match.lineups.home.starting_xi}
            home_regulars = self._recent_starters.get(home_team, set())
            features["regulars_missing_home"] = len(home_regulars - home_xi)
            
            away_xi = {p.id for p in match.lineups.away.starting_xi}
            away_regulars = self._recent_starters.get(away_team, set())
            features["regulars_missing_away"] = len(away_regulars - away_xi)
        else:
            features["regulars_missing_home"] = 0
            features["regulars_missing_away"] = 0
        
        return features
