"""
Advanced statistics feature builder.

Computes rolling averages for:
- Expected Goals (xG) For/Against
- Shots / Shots on Target
- xG Performance (Goals - xG)
"""

from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import numpy as np

from stavki.features import FeatureBuilder
from stavki.data.schemas import Match

class AdvancedFeatureBuilder(FeatureBuilder):
    """
    Computes rolling advanced statistics.
    """
    name = "advanced"
    
    def __init__(self, window: int = 5):
        self.window = window
        
    def _get_team_matches(
        self,
        team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> List[Match]:
        """Get recent matches for a team."""
        team_matches = [
            m for m in matches 
            if (m.home_team.normalized_name == team or 
                m.away_team.normalized_name == team)
            and m.is_completed
            and m.stats is not None # Only use matches with stats
        ]
        
        if as_of:
            team_matches = [m for m in team_matches if m.commence_time < as_of]
            
        team_matches.sort(key=lambda m: m.commence_time)
        return team_matches[-self.window:]
        
    def compute_for_team(
        self,
        team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Compute rolling stats for a team."""
        recent = self._get_team_matches(team, matches, as_of)
        
        if not recent:
            return {
                "xg_for": 0.0, "xg_against": 0.0,
                "shots_for": 0.0, "shots_against": 0.0,
                "sot_for": 0.0, "sot_against": 0.0,
                "xg_perf": 0.0
            }
            
        xg_for = []
        xg_against = []
        shots_for = []
        shots_against = []
        sot_for = []
        sot_against = []
        goals = []
        
        for m in recent:
            if not m.stats:
                continue
                
            is_home = m.home_team.normalized_name == team
            
            # xG
            xf = m.stats.xg_home if is_home else m.stats.xg_away
            xa = m.stats.xg_away if is_home else m.stats.xg_home
            xg_for.append(xf or 0.0)
            xg_against.append(xa or 0.0)
            
            # Shots
            sf = m.stats.shots_home if is_home else m.stats.shots_away
            sa = m.stats.shots_away if is_home else m.stats.shots_home
            shots_for.append(sf or 0)
            shots_against.append(sa or 0)
            
            # SOT
            stf = m.stats.shots_on_target_home if is_home else m.stats.shots_on_target_away
            sta = m.stats.shots_on_target_away if is_home else m.stats.shots_on_target_home
            sot_for.append(stf or 0)
            sot_against.append(sta or 0)
            
            # Goals (for performance diff)
            g = m.home_score if is_home else m.away_score
            goals.append(g or 0)
            
        n = len(xg_for)
        if n == 0:
            return {
                "xg_for": 0.0, "xg_against": 0.0,
                "shots_for": 0.0, "shots_against": 0.0,
                "sot_for": 0.0, "sot_against": 0.0,
                "xg_perf": 0.0
            }
            
        avg_xg_for = sum(xg_for) / n
        avg_xg_against = sum(xg_against) / n
        avg_goals = sum(goals) / n
        
        return {
            "xg_for": avg_xg_for,
            "xg_against": avg_xg_against,
            "xg_diff": avg_xg_for - avg_xg_against,
            "shots_for": sum(shots_for) / n,
            "shots_against": sum(shots_against) / n,
            "sot_for": sum(sot_for) / n,
            "sot_against": sum(sot_against) / n,
            "xg_perf": avg_goals - avg_xg_for # +ve = overperforming/lucky, -ve = underperforming/unlucky
        }

    def build(
        self,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Build features for all matches."""
        # Filter matches to build features FOR (not FROM - compute_for_team handles FROM)
        target_matches = matches
        if as_of:
            target_matches = [m for m in matches if m.commence_time < as_of] # Usually we build for historical + upcoming
            
        if not target_matches:
            return pd.DataFrame()
            
        # Optimization: Pre-compute team stats per date? 
        # For simplicity, sticking to loop, but caching could be added if needed.
        
        rows = []
        for m in target_matches:
            # Stats known BEFORE the match
            home_stats = self.compute_for_team(m.home_team.normalized_name, matches, as_of=m.commence_time)
            away_stats = self.compute_for_team(m.away_team.normalized_name, matches, as_of=m.commence_time)
            
            row = {"match_id": m.id}
            
            # Flatten home
            for k, v in home_stats.items():
                row[f"home_{k}"] = v
                
            # Flatten away
            for k, v in away_stats.items():
                row[f"away_{k}"] = v
                
            rows.append(row)
            
        return pd.DataFrame(rows)
