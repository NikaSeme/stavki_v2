"""
Base interfaces for feature builders.

All feature builders implement FeatureBuilder ABC.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import logging

from stavki.data.schemas import Match, TeamFeatures


logger = logging.getLogger(__name__)


class FeatureBuilder(ABC):
    """
    Base class for all feature builders.
    
    Feature builders compute features from historical match data.
    They must be time-aware to prevent data leakage (future data).
    
    Usage:
        builder = EloBuilder(k_factor=32)
        features = builder.build(historical_matches, as_of=match_date)
    """
    
    name: str = "base"
    
    @abstractmethod
    def build(
        self,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Build features from match history.
        
        Args:
            matches: List of historical matches with results
            as_of: Only use data before this time (prevents leakage)
            
        Returns:
            DataFrame with computed features
        """
        pass
    
    def build_for_match(
        self,
        home_team: str,
        away_team: str,
        historical_matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Build features for a specific upcoming match.
        
        Default implementation builds full dataset and extracts last row.
        Override for efficiency if needed.
        """
        raise NotImplementedError("Subclasses should implement build_for_match")


class TeamFeatureBuilder(FeatureBuilder):
    """
    Base class for team-level feature builders.
    
    Computes features per team, then merges for home/away.
    """
    
    @abstractmethod
    def compute_for_team(
        self,
        team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> float:
        """Compute feature value for a single team."""
        pass
    
    def build(
        self,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Build home and away features."""
        # Filter matches by time
        if as_of:
            matches = [m for m in matches if m.commence_time < as_of and m.is_completed]
        else:
            matches = [m for m in matches if m.is_completed]
        
        if not matches:
            return pd.DataFrame()
        
        # Compute per team
        teams = set()
        for m in matches:
            teams.add(m.home_team.normalized_name)
            teams.add(m.away_team.normalized_name)
        
        team_values = {}
        for team in teams:
            team_values[team] = self.compute_for_team(team, matches, as_of)
        
        # Build rows
        rows = []
        for m in matches:
            rows.append({
                'match_id': m.id,
                f'{self.name}_home': team_values.get(m.home_team.normalized_name, 0),
                f'{self.name}_away': team_values.get(m.away_team.normalized_name, 0),
            })
        
        return pd.DataFrame(rows)


class RollingFeatureBuilder(TeamFeatureBuilder):
    """
    Base class for rolling window features.
    
    Computes statistics over last N matches.
    """
    
    def __init__(self, window: int = 5):
        self.window = window
    
    @abstractmethod
    def aggregate(self, recent_matches: List[Match], team: str) -> float:
        """Aggregate feature from recent matches."""
        pass
    
    def compute_for_team(
        self,
        team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> float:
        """Get last N matches for team and aggregate."""
        # Get matches involving this team
        team_matches = [
            m for m in matches 
            if (m.home_team.normalized_name == team or 
                m.away_team.normalized_name == team)
            and m.is_completed
        ]
        
        # Filter by time
        if as_of:
            team_matches = [m for m in team_matches if m.commence_time < as_of]
        
        # Sort by date, take last N
        team_matches.sort(key=lambda m: m.commence_time)
        recent = team_matches[-self.window:] if len(team_matches) >= self.window else team_matches
        
        if not recent:
            return 0.0
        
        return self.aggregate(recent, team)
