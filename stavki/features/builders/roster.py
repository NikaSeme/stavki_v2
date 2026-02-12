"""
Roster feature builder.

Computes features based on team lineups:
- Lineup Stability (Are these regular starters?)
- Squad Rotation (How many changes from last match?)
- Experience (Aggregate matches played)
"""

from typing import List, Optional, Dict, Set
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict

from stavki.features import FeatureBuilder
from stavki.data.schemas import Match

class RosterFeatureBuilder(FeatureBuilder):
    name = "roster"
    
    def __init__(self, window: int = 20):
        self.window = window
        self.player_starts: Dict[str, List[datetime]] = defaultdict(list)
        
    def fit(self, matches: List[Match]) -> None:
        """
        Fit the builder by indexing player history.
        """
        # Index: player_id -> list of start timestamps (sorted)
        self.player_starts: Dict[str, List[datetime]] = defaultdict(list)
        
        for m in matches:
            if not m.is_completed or not m.lineups:
                continue
                
            # Home
            if m.lineups.home and m.lineups.home.starting_xi:
                for p in m.lineups.home.starting_xi:
                    self.player_starts[p.id].append(m.commence_time)
            
            # Away
            if m.lineups.away and m.lineups.away.starting_xi:
                for p in m.lineups.away.starting_xi:
                    self.player_starts[p.id].append(m.commence_time)
                    
        # Sort all lists (just in case)
        for pid in self.player_starts:
            self.player_starts[pid].sort()
            
    def build(
        self,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Build features for a list of matches.
        """
        if not hasattr(self, 'player_starts'):
            self.fit(matches) # Self-fit if not already fitted
            
        rows = []
        for m in matches:
            if as_of and m.commence_time >= as_of:
                continue
                
            feats = self.get_features(m, as_of)
            feats["match_id"] = m.id
            rows.append(feats)
            
        return pd.DataFrame(rows)
            
    def get_features(
        self,
        match: Match,
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Get roster features for a specific match.
        """
        # If no lineups available for this match (e.g. far future), return defaults
        if not match.lineups:
             return {
                "roster_regularity_home": 0.5, "roster_experience_home": 0.0,
                "roster_regularity_away": 0.5, "roster_experience_away": 0.0
             }
             
        cutoff = as_of or match.commence_time
        
        home_feats = self._calc_lineup_strength(match.lineups.home, cutoff)
        away_feats = self._calc_lineup_strength(match.lineups.away, cutoff)
        
        return {
            "roster_regularity_home": home_feats["regularity"],
            "roster_experience_home": home_feats["experience"],
            "roster_regularity_away": away_feats["regularity"],
            "roster_experience_away": away_feats["experience"],
        }

    def _calc_lineup_strength(self, lineup: object, cutoff: datetime) -> Dict[str, float]:
        if not lineup or not lineup.starting_xi:
            return {"regularity": 0.5, "experience": 0.0}
            
        starters = [p.id for p in lineup.starting_xi]
        if not starters:
             return {"regularity": 0.5, "experience": 0.0}
             
        total_regularity = 0.0
        total_experience = 0.0
        
        import math
        
        for pid in starters:
            starts = self.player_starts.get(pid, [])
            
            # Filter history strictly BEFORE cutoff
            # We can use bisect for speed if needed, but linear scan of a player's history is tiny
            past_starts = [t for t in starts if t < cutoff]
            
            # Regularity: Starts in last 90 days
            recent_starts = [t for t in past_starts if t > cutoff - timedelta(days=90)]
            regularity = min(len(recent_starts) / 10.0, 1.0)
            total_regularity += regularity
            
            # Experience: Total starts
            experience = math.log1p(len(past_starts))
            total_experience += experience
            
        return {
            "regularity": total_regularity / len(starters),
            "experience": total_experience
        }

