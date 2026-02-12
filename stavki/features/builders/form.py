"""
Form and goals feature builders.

Rolling statistics over last N matches:
- Form points (W=3, D=1, L=0)
- Goals scored/conceded
- Win/loss streaks
- Home/away specific form
"""

from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

from stavki.data.schemas import Match, Outcome

import logging

logger = logging.getLogger(__name__)


@dataclass
class FormStats:
    """Form statistics for a team over N matches."""
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    
    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws
    
    @property
    def matches(self) -> int:
        return self.wins + self.draws + self.losses
    
    @property
    def points_per_game(self) -> float:
        if self.matches == 0:
            return 0
        return self.points / self.matches
    
    @property
    def goals_scored_avg(self) -> float:
        if self.matches == 0:
            return 0
        return self.goals_scored / self.matches
    
    @property
    def goals_conceded_avg(self) -> float:
        if self.matches == 0:
            return 0
        return self.goals_conceded / self.matches
    
    @property
    def goal_diff(self) -> int:
        return self.goals_scored - self.goals_conceded


class FormCalculator:
    """
    Calculate form statistics for teams.
    
    Features:
    - Overall form (last N matches)
    - Home/away specific form
    - Streaks (win streak, unbeaten streak)
    - Weighted form (recent matches count more)
    """
    
    def __init__(self, window: int = 5, weighted: bool = False):
        self.window = window
        self.weighted = weighted
    
    def _get_team_matches(
        self,
        team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None,
        home_only: bool = False,
        away_only: bool = False
    ) -> List[Match]:
        """Get matches involving team, optionally filtered."""
        result = []
        
        for m in matches:
            if not m.is_completed:
                continue
            if as_of and m.commence_time >= as_of:
                continue
            
            is_home = m.home_team.normalized_name == team
            is_away = m.away_team.normalized_name == team
            
            if home_only and not is_home:
                continue
            if away_only and not is_away:
                continue
            
            if is_home or is_away:
                result.append(m)
        
        # Sort by date descending and take last N
        result.sort(key=lambda x: x.commence_time, reverse=True)
        return result[:self.window]
    
    def _match_result_for_team(self, match: Match, team: str) -> str:
        """Get result (W/D/L) for specific team."""
        is_home = match.home_team.normalized_name == team
        
        if match.result == Outcome.DRAW:
            return "D"
        elif match.result == Outcome.HOME:
            return "W" if is_home else "L"
        else:  # AWAY
            return "L" if is_home else "W"
    
    def _goals_for_team(self, match: Match, team: str) -> tuple:
        """Get (goals_scored, goals_conceded) for team."""
        is_home = match.home_team.normalized_name == team
        
        if is_home:
            return match.home_score, match.away_score
        else:
            return match.away_score, match.home_score
    
    def calculate(
        self,
        team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None,
        home_only: bool = False,
        away_only: bool = False
    ) -> FormStats:
        """Calculate form statistics for a team."""
        team_matches = self._get_team_matches(team, matches, as_of, home_only, away_only)
        
        stats = FormStats()
        
        for m in team_matches:
            result = self._match_result_for_team(m, team)
            goals_for, goals_against = self._goals_for_team(m, team)
            
            if result == "W":
                stats.wins += 1
            elif result == "D":
                stats.draws += 1
            else:
                stats.losses += 1
            
            stats.goals_scored += goals_for
            stats.goals_conceded += goals_against
        
        return stats
    
    def get_streak(
        self,
        team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None,
        streak_type: str = "win"  # "win", "unbeaten", "clean_sheet"
    ) -> int:
        """Calculate current streak length."""
        team_matches = self._get_team_matches(team, matches, as_of)
        
        # Matches are already sorted most recent first
        streak = 0
        
        for m in team_matches:
            result = self._match_result_for_team(m, team)
            goals_for, goals_against = self._goals_for_team(m, team)
            
            if streak_type == "win":
                if result == "W":
                    streak += 1
                else:
                    break
            elif streak_type == "unbeaten":
                if result in ["W", "D"]:
                    streak += 1
                else:
                    break
            elif streak_type == "clean_sheet":
                if goals_against == 0:
                    streak += 1
                else:
                    break
        
        return streak


class FormBuilder:
    """
    Feature builder using FormCalculator.
    
    Returns form-related features for matches.
    """
    
    def __init__(self, window: int = 5):
        self.calculator = FormCalculator(window=window)
    
    def get_features(
        self,
        home_team: str,
        away_team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get form features for a match."""
        # Overall form
        home_form = self.calculator.calculate(home_team, matches, as_of)
        away_form = self.calculator.calculate(away_team, matches, as_of)
        
        # Home/away specific form
        home_at_home = self.calculator.calculate(home_team, matches, as_of, home_only=True)
        away_at_away = self.calculator.calculate(away_team, matches, as_of, away_only=True)
        
        # Streaks
        home_win_streak = self.calculator.get_streak(home_team, matches, as_of, "win")
        away_win_streak = self.calculator.get_streak(away_team, matches, as_of, "win")
        home_unbeaten = self.calculator.get_streak(home_team, matches, as_of, "unbeaten")
        away_unbeaten = self.calculator.get_streak(away_team, matches, as_of, "unbeaten")
        
        return {
            # Form points
            "form_home": home_form.points,
            "form_away": away_form.points,
            "form_diff": home_form.points - away_form.points,
            "form_ppg_home": home_form.points_per_game,
            "form_ppg_away": away_form.points_per_game,
            
            # Home/Away specific
            "form_home_at_home": home_at_home.points,
            "form_away_at_away": away_at_away.points,
            
            # Goals
            "goals_scored_home": home_form.goals_scored_avg,
            "goals_conceded_home": home_form.goals_conceded_avg, 
            "goals_scored_away": away_form.goals_scored_avg,
            "goals_conceded_away": away_form.goals_conceded_avg,
            "goal_diff_form_home": home_form.goal_diff,
            "goal_diff_form_away": away_form.goal_diff,
            
            # Streaks
            "win_streak_home": home_win_streak,
            "win_streak_away": away_win_streak,
            "unbeaten_streak_home": home_unbeaten,
            "unbeaten_streak_away": away_unbeaten,
        }


class GoalsBuilder:
    """
    Goal-specific feature builder.
    
    More detailed goal statistics.
    """
    
    def __init__(self, window: int = 10):
        self.window = window
        self.calculator = FormCalculator(window=window)
    
    def get_features(
        self,
        home_team: str,
        away_team: str,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get goal-related features."""
        home_stats = self.calculator.calculate(home_team, matches, as_of)
        away_stats = self.calculator.calculate(away_team, matches, as_of)
        
        # Attack strength relative to average (1.4 goals per game league avg)
        league_avg = 1.4
        
        home_attack = home_stats.goals_scored_avg / league_avg if league_avg > 0 else 1
        home_defense = home_stats.goals_conceded_avg / league_avg if league_avg > 0 else 1
        away_attack = away_stats.goals_scored_avg / league_avg if league_avg > 0 else 1
        away_defense = away_stats.goals_conceded_avg / league_avg if league_avg > 0 else 1
        
        return {
            "attack_strength_home": home_attack,
            "defense_strength_home": home_defense,
            "attack_strength_away": away_attack,
            "defense_strength_away": away_defense,
            
            # Expected goals for Poisson model
            "expected_home_goals": home_attack * away_defense * 1.5,  # Home boost
            "expected_away_goals": away_attack * home_defense * 1.2,
        }
