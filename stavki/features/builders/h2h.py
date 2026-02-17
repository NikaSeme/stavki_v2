"""
Head-to-head (H2H) feature builder.

Computes statistics from historical meetings between two teams.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime

from stavki.data.schemas import Match, Outcome, H2HFeatures


class H2HBuilder:
    """
    Head-to-head feature builder.
    
    Features:
    - Historical record (home wins, draws, away wins)
    - Recent H2H results
    - Average goals in meetings
    - Trend detection (which team is improving)
    """
    
    def __init__(self, max_meetings: int = 10, recent_window: int = 5):
        self.max_meetings = max_meetings
        self.recent_window = recent_window
    
    def _get_meetings(
        self,
        home_team: str,
        away_team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> List[Match]:
        """Get historical meetings between teams."""
        meetings = []
        
        for m in matches:
            if not m.is_completed:
                continue
            if as_of and m.commence_time >= as_of:
                continue
            
            # Match if these teams played (in either configuration)
            teams = {m.home_team.normalized_name, m.away_team.normalized_name}
            if home_team in teams and away_team in teams:
                meetings.append(m)
        
        # Sort by date, most recent first
        meetings.sort(key=lambda x: x.commence_time, reverse=True)
        return meetings[:self.max_meetings]
    
    def calculate(
        self,
        home_team: str,
        away_team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> H2HFeatures:
        """Calculate H2H statistics."""
        meetings = self._get_meetings(home_team, away_team, matches, as_of)
        
        # Counters - from perspective of current home team
        # (but in H2H, each meeting could have either team at home)
        home_wins = 0
        draws = 0  
        away_wins = 0
        total_home_goals = 0
        total_away_goals = 0
        
        # Recent meetings (last 5)
        recent_home_wins = 0
        recent_draws = 0
        recent_away_wins = 0
        
        for i, m in enumerate(meetings):
            is_home_at_home = m.home_team.normalized_name == home_team
            
            # Goals from current home team's perspective
            if is_home_at_home:
                my_goals = m.home_score
                their_goals = m.away_score
            else:
                my_goals = m.away_score
                their_goals = m.home_score
            
            total_home_goals += my_goals
            total_away_goals += their_goals
            
            # Result from current home team's perspective
            if my_goals > their_goals:
                home_wins += 1
                if i < self.recent_window:
                    recent_home_wins += 1
            elif my_goals < their_goals:
                away_wins += 1
                if i < self.recent_window:
                    recent_away_wins += 1
            else:
                draws += 1
                if i < self.recent_window:
                    recent_draws += 1
        
        total = len(meetings)
        
        return H2HFeatures(
            home_team=home_team,
            away_team=away_team,
            computed_at=datetime.utcnow(),
            total_matches=total,
            home_wins=home_wins,
            draws=draws,
            away_wins=away_wins,
            recent_home_wins=recent_home_wins,
            recent_draws=recent_draws,
            recent_away_wins=recent_away_wins,
            avg_total_goals=(total_home_goals + total_away_goals) / total if total > 0 else 0.0,
            avg_home_goals=total_home_goals / total if total > 0 else 0.0,
            avg_away_goals=total_away_goals / total if total > 0 else 0.0,
            home_winning_trend=recent_home_wins > (home_wins / 2 if home_wins > 0 else 0),
            away_winning_trend=recent_away_wins > (away_wins / 2 if away_wins > 0 else 0),
        )
    
    def get_features(
        self,
        home_team: str,
        away_team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get H2H features for a match."""
        h2h = self.calculate(home_team, away_team, matches, as_of)
        
        return {
            "h2h_matches": h2h.total_matches,
            "h2h_home_wins": h2h.home_wins,
            "h2h_draws": h2h.draws,
            "h2h_away_wins": h2h.away_wins,
            "h2h_home_win_rate": h2h.home_win_rate,
            "h2h_draw_rate": h2h.draw_rate,
            "h2h_avg_goals": h2h.avg_total_goals,
            "h2h_home_advantage": (h2h.home_wins - h2h.away_wins) / max(h2h.total_matches, 1),
            "h2h_recent_momentum": h2h.recent_home_wins - h2h.recent_away_wins,
        }
