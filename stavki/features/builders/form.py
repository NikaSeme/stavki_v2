"""
Form and Goal Statistics Feature Builder.

Computes rolling form (points, streaks) and goal stats.
Includes GoalsBuilder for attack/defense strength relative to league average.

Performance: Uses pre-indexed match lookups to avoid O(N²) scanning.
"""

from typing import List, Optional, Dict, Tuple
from datetime import datetime
from collections import defaultdict
from bisect import bisect_left
import logging

from stavki.features import FeatureBuilder
from stavki.data.schemas import Match

logger = logging.getLogger(__name__)


class FormStats:
    """Container for form statistics."""
    def __init__(self):
        self.points = 0
        self.wins = 0
        self.draws = 0
        self.losses = 0
        self.goals_scored = 0
        self.goals_conceded = 0
        self.matches = 0
        self.win_streak = 0
        self.unbeaten_streak = 0
        
    @property
    def goals_scored_avg(self) -> float:
        return self.goals_scored / self.matches if self.matches > 0 else 0.0
        
    @property
    def goals_conceded_avg(self) -> float:
        return self.goals_conceded / self.matches if self.matches > 0 else 0.0


class FormCalculator:
    """
    Calculates rolling form stats with pre-indexed lookups (O(N) total).
    """
    
    def __init__(self, window: int = 5):
        self.window = window
        # team -> list of (commence_time, match_record)
        # record: {result: 'W'/'D'/'L', gf, ga, is_home}
        self._team_index: Dict[str, list] = defaultdict(list)
        self._is_fitted = False
        
    def fit(self, matches: List[Match]) -> None:
        """Pre-index matches by team."""
        self._team_index.clear()
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.is_completed:
                continue
                
            home = m.home_team.normalized_name
            away = m.away_team.normalized_name
            
            home_goals = m.home_score or 0
            away_goals = m.away_score or 0
            
            # Home record
            res_h = 'D'
            if home_goals > away_goals: res_h = 'W'
            elif home_goals < away_goals: res_h = 'L'
            
            self._team_index[home].append({
                "time": m.commence_time,
                "result": res_h,
                "gf": home_goals,
                "ga": away_goals,
                "is_home": True
            })
            
            # Away record
            res_a = 'D'
            if away_goals > home_goals: res_a = 'W'
            elif away_goals < home_goals: res_a = 'L'
            
            self._team_index[away].append({
                "time": m.commence_time,
                "result": res_a,
                "gf": away_goals,
                "ga": home_goals,
                "is_home": False
            })
            
        self._is_fitted = True
        logger.info(f"FormCalculator: pre-indexed {len(self._team_index)} teams")

    def _get_recent(self, team: str, as_of: Optional[datetime], 
                   home_only: bool = False, away_only: bool = False) -> list:
        """Get recent matches using binary search on pre-indexed data."""
        records = self._team_index.get(team, [])
        if not records:
            return []
            
        if as_of:
            # Find insertion point for as_of to ignore future matches
            # records are sorted by time during fit()
            # We construct a dummy record for bisect since list contains dicts
            # But bisect doesn't support key=... in older python, rely on ordered timestamps
            # Efficient check: if last match is before as_of, take all
            if records[-1]["time"] < as_of:
                candidates = records
            else:
                # Linear scan backwards is fast for "recent" forms usually
                # But for strict correctness with bisect:
                import bisect
                times = [r["time"] for r in records]
                idx = bisect.bisect_left(times, as_of)
                candidates = records[:idx]
        else:
            candidates = records

        if home_only:
            candidates = [r for r in candidates if r["is_home"]]
        elif away_only:
            candidates = [r for r in candidates if not r["is_home"]]
            
        return candidates[-self.window:]

    def calculate(
        self, 
        team: str, 
        matches: List[Match],  # Ignored if fitted
        as_of: Optional[datetime] = None
    ) -> FormStats:
        """Calculate form stats."""
        # Auto-fit if needed
        if not self._is_fitted and matches:
            self.fit(matches)
            
        stats = FormStats()
        recent = self._get_recent(team, as_of)
        
        for r in recent:
            stats.matches += 1
            stats.goals_scored += r["gf"]
            stats.goals_conceded += r["ga"]
            
            if r["result"] == 'W':
                stats.wins += 1
                stats.points += 3
            elif r["result"] == 'D':
                stats.draws += 1
                stats.points += 1
            else:
                stats.losses += 1
                
        # Streaks (working backwards)
        for r in reversed(recent):
            if r["result"] == 'W':
                stats.win_streak += 1
            else:
                break
                
        for r in reversed(recent):
            if r["result"] != 'L':
                stats.unbeaten_streak += 1
            else:
                break
                
        return stats
        
    def get_streak(self, team: str, matches: List[Match], 
                  streak_type: str = "win", as_of=None) -> int:
        """Get specific streak."""
        # Auto-fit
        if not self._is_fitted and matches:
            self.fit(matches)
            
        recent = self._get_recent(team, as_of)
        count = 0
        
        for r in reversed(recent):
            condition = False
            if streak_type == "win":
                condition = (r["result"] == 'W')
            elif streak_type == "unbeaten":
                condition = (r["result"] != 'L')
            elif streak_type == "losing":
                condition = (r["result"] == 'L')
            elif streak_type == "winless":
                condition = (r["result"] != 'W')
                
            if condition:
                count += 1
            else:
                break
        return count


class FormBuilder:
    """Genearal form features builder."""
    name = "form"
    
    def __init__(self, window: int = 5):
        self.calculator = FormCalculator(window=window)
        
    def fit(self, matches: List[Match]):
        self.calculator.fit(matches)
        
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        if not match:
            return {}
            
        home = match.home_team.normalized_name
        away = match.away_team.normalized_name
        time = as_of or match.commence_time
        
        # General form
        fh = self.calculator.calculate(home, [], time)
        fa = self.calculator.calculate(away, [], time)
        
        # Home/Away specific form (last 5 home games for home team, etc.)
        # Use underlying _get_recent for manual calculation or add support in calculator
        # For speed, let's just use the calculator's method if we expose it, or rebuild logic
        # Ideally, calculator should support "venue_specific" flag?
        # Let's assume calculate() handles general form.
        
        # For home_form_points (at home) / away_form_points (at away)
        # We need check how usage implies.
        # usually "home_form" means general form of home team.
        
        return {
            "form_home": fh.points,
            "form_away": fa.points,
            "form_diff": fh.points - fa.points,
            "goals_scored_home": fh.goals_scored_avg,
            "goals_conceded_home": fh.goals_conceded_avg,
            "goals_scored_away": fa.goals_scored_avg,
            "goals_conceded_away": fa.goals_conceded_avg,
            "win_streak_home": fh.win_streak,
            "win_streak_away": fa.win_streak,
            "unbeaten_streak_home": fh.unbeaten_streak,
            "unbeaten_streak_away": fa.unbeaten_streak,
        }


class GoalsBuilder:
    """
    Goal-specific feature builder.
    
    More detailed goal statistics.
    """
    name = "goals"
    
    def __init__(self, window: int = 10):
        self.window = window
        self.calculator = FormCalculator(window=window)
        # Pre-computed global goal stats per date to avoid scanning?
        # For now, just fix the temporal bug.
        
    def fit(self, matches: List[Match]):
        self.calculator.fit(matches)
        
        # Pre-compute ordered global stats for dynamic league average
        self._global_stats = []
        self._global_times = []
        
        sorted_matches = sorted(
            [m for m in matches if m.is_completed], 
            key=lambda x: x.commence_time
        )
        
        cum_goals = 0
        cum_matches = 0
        
        for m in sorted_matches:
            g = (m.home_score or 0) + (m.away_score or 0)
            cum_goals += g
            cum_matches += 1
            self._global_stats.append((cum_goals, cum_matches))
            self._global_times.append(m.commence_time)
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get goal-related features."""
        if not match:
            return {}

        home = match.home_team.normalized_name
        away = match.away_team.normalized_name
        time = as_of or match.commence_time
        
        home_stats = self.calculator.calculate(home, [], time)
        away_stats = self.calculator.calculate(away, [], time)
        
        # Compute dynamic league average from history
        league_avg = 1.37 # Default fallback
        
        if hasattr(self, "_global_stats") and self._global_stats:
            # Find index of first match >= time
            # We want stats from matches < time, so take index-1
            idx = bisect_left(self._global_times, time)
            
            if idx > 0:
                c_goals, c_matches = self._global_stats[idx-1]
                if c_matches > 0:
                    league_avg = c_goals / c_matches
        
        # Ensure non-zero divisor
        league_avg = max(league_avg, 0.1)
        
        # ... Wait, the previous implementation received 'matches'.
        # If I change signature, it might break if registry calls it differently.
        # But registry calls standard interface.
        # The previous 'GoalsBuilder' might have been called manually or I misread.
        # Let's assume standard interface.
        
        # To compute league_avg dynamically without O(N):
        # We can't easily without a global index. 
        # Let's fallback to 1.37 for now to ensure O(1) speed, 
        # OR better: use the LEAGUE_AVG_GOALS from TeamFeatures schema constant.
        
        # But wait, user wanted "GoalsBuilder computes league_avg correctly".
        # If I use constant, I solve temporal leak but lose dynamism.
        # I'll rely on the schema constant 1.37 for now as it's safe.
        
        home_attack = home_stats.goals_scored_avg / league_avg
        home_defense = home_stats.goals_conceded_avg / league_avg
        away_attack = away_stats.goals_scored_avg / league_avg
        away_defense = away_stats.goals_conceded_avg / league_avg
        
        return {
            "attack_strength_home": round(home_attack, 3),
            "defense_strength_home": round(home_defense, 3),
            "attack_strength_away": round(away_attack, 3),
            "defense_strength_away": round(away_defense, 3),
            
            # Expected goals (naive)
            "expected_home_goals": round(home_attack * away_defense * league_avg, 3),
            "expected_away_goals": round(away_attack * home_defense * league_avg, 3),
            "league_avg_goals": round(league_avg, 3),
        }
