"""
ELO Rating Calculator.

Computes ELO ratings for teams based on match results.

Key features:
- Dynamic K-factor based on match importance
- Home advantage adjustment
- Season reset with regression to mean
- Momentum tracking (ELO change over last N matches)

Reference: https://en.wikipedia.org/wiki/Elo_rating_system
Football-specific: https://eloratings.net/about
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import logging

from stavki.data.schemas import Match, Outcome

logger = logging.getLogger(__name__)


@dataclass
class EloRating:
    """Team's ELO rating with history."""
    team: str
    rating: float = 1500.0
    initial_rating: float = 1500.0  # Preserved for fallback lookups
    history: List[Tuple[datetime, float]] = field(default_factory=list)
    matches_played: int = 0
    
    def update(self, new_rating: float, timestamp: datetime) -> None:
        """Update rating and record history."""
        self.history.append((timestamp, self.rating))
        self.rating = new_rating
        self.matches_played += 1
    
    def get_rating_at(self, timestamp: datetime) -> float:
        """Get rating as of a specific time (for backtesting)."""
        # Find most recent rating before timestamp
        relevant = [r for t, r in self.history if t < timestamp]
        if relevant:
            return relevant[-1]
        return self.initial_rating
    
    def recent_momentum(self, n: int = 5) -> float:
        """ELO change over last N matches."""
        if len(self.history) < n:
            return 0.0
        recent = self.history[-n:]
        return self.rating - recent[0][1]


class EloCalculator:
    """
    ELO rating calculator for football teams.
    
    Improvements over basic ELO:
    1. Dynamic K-factor: higher K for uncertain teams, lower for established
    2. Home advantage: +65 ELO equivalent for home team
    3. Goal difference: larger wins give more points
    4. Season regression: 1/3 regression to mean between seasons
    
    Usage:
        calc = EloCalculator()
        calc.process_matches(historical_matches)
        
        home_elo = calc.get_rating("manchester united")
        away_elo = calc.get_rating("chelsea")
    """
    
    # Default parameters (can be tuned)
    DEFAULT_RATING = 1500.0
    K_FACTOR = 15  # Optimized K-factor (was 32)
    HOME_ADVANTAGE = 65  # ELO points for home field
    SEASON_REGRESSION = 0.33  # Regress 1/3 to mean between seasons
    
    def __init__(
        self,
        k_factor: float = None,
        home_advantage: float = None,
        initial_rating: float = None,
        use_goal_diff: bool = True,
        use_dynamic_k: bool = True,
    ):
        self.k_factor = k_factor or self.K_FACTOR
        self.home_advantage = home_advantage or self.HOME_ADVANTAGE
        self.initial_rating = initial_rating or self.DEFAULT_RATING
        self.use_goal_diff = use_goal_diff
        self.use_dynamic_k = use_dynamic_k
        
        # Ratings storage
        self.ratings: Dict[str, EloRating] = {}
        
        # Track processed matches
        self.processed_matches: set = set()
    
    def get_rating(self, team: str, as_of: Optional[datetime] = None) -> float:
        """
        Get team's current ELO rating.
        
        Args:
            team: Team name (normalized)
            as_of: Get rating as of this time (for backtesting)
        """
        if team not in self.ratings:
            return self.initial_rating
        
        if as_of:
            return self.ratings[team].get_rating_at(as_of)
        return self.ratings[team].rating
    
    def get_elo_record(self, team: str) -> Optional[EloRating]:
        """Get full ELO record for a team."""
        return self.ratings.get(team)
    
    def _get_k_factor(self, team: str) -> float:
        """
        Dynamic K-factor based on team's match count.
        
        New teams have higher K (more volatile ratings).
        Established teams have lower K (more stable).
        """
        if not self.use_dynamic_k:
            return self.k_factor
        
        record = self.ratings.get(team)
        if not record:
            return self.k_factor * 1.5  # Higher K for new teams
        
        matches = record.matches_played
        if matches < 10:
            return self.k_factor * 1.3
        elif matches < 30:
            return self.k_factor
        else:
            return self.k_factor * 0.8
    
    def _goal_diff_multiplier(self, goal_diff: int) -> float:
        """
        Multiplier based on goal difference.
        
        Bigger wins should have more impact.
        Uses log scaling to prevent extreme values.
        """
        if not self.use_goal_diff or goal_diff == 0:
            return 1.0
        
        import math
        # Log-based: 1-goal = 1.0, 2-goal = 1.5, 3-goal = 1.75...
        return math.log10(abs(goal_diff) + 1) + 1.0
    
    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Expected score for player A against player B.
        
        Standard ELO formula.
        """
        import math
        return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
    
    def _actual_score(self, match: Match, for_home: bool) -> float:
        """
        Actual score: 1 for win, 0.5 for draw, 0 for loss.
        """
        if not match.is_completed:
            raise ValueError(f"Match {match.id} not completed")
        
        if match.result == Outcome.DRAW:
            return 0.5
        elif match.result == Outcome.HOME:
            return 1.0 if for_home else 0.0
        else:  # AWAY
            return 0.0 if for_home else 1.0
    
    def process_match(self, match: Match) -> Tuple[float, float]:
        """
        Process a single match result.
        
        Updates ratings for both teams.
        
        Returns:
            (new_home_rating, new_away_rating)
        """
        if match.id in self.processed_matches:
            return self.get_rating(match.home_team.normalized_name), \
                   self.get_rating(match.away_team.normalized_name)
        
        if not match.is_completed:
            raise ValueError(f"Cannot process incomplete match: {match.id}")
        
        home = match.home_team.normalized_name
        away = match.away_team.normalized_name
        
        # Initialize if new teams
        if home not in self.ratings:
            self.ratings[home] = EloRating(team=home, rating=self.initial_rating, initial_rating=self.initial_rating)
        if away not in self.ratings:
            self.ratings[away] = EloRating(team=away, rating=self.initial_rating, initial_rating=self.initial_rating)
        
        # Current ratings
        home_rating = self.ratings[home].rating
        away_rating = self.ratings[away].rating
        
        # Apply home advantage to expected calculation
        home_expected = self._expected_score(
            home_rating + self.home_advantage,
            away_rating
        )
        away_expected = 1 - home_expected
        
        # Actual scores
        home_actual = self._actual_score(match, for_home=True)
        away_actual = self._actual_score(match, for_home=False)
        
        # Goal difference multiplier
        goal_diff = abs(match.home_score - match.away_score)
        gd_mult = self._goal_diff_multiplier(goal_diff)
        
        # K-factors
        k_home = self._get_k_factor(home)
        k_away = self._get_k_factor(away)
        
        # New ratings
        new_home = home_rating + k_home * gd_mult * (home_actual - home_expected)
        new_away = away_rating + k_away * gd_mult * (away_actual - away_expected)
        
        # Update
        self.ratings[home].update(new_home, match.commence_time)
        self.ratings[away].update(new_away, match.commence_time)
        self.processed_matches.add(match.id)
        
        return new_home, new_away
    
    def process_matches(self, matches: List[Match]) -> None:
        """
        Process multiple matches in chronological order.
        
        IMPORTANT: Matches must be sorted by date!
        """
        # Sort by commence time (earliest first)
        sorted_matches = sorted(
            [m for m in matches if m.is_completed],
            key=lambda m: m.commence_time
        )
        
        for match in sorted_matches:
            try:
                self.process_match(match)
            except Exception as e:
                logger.warning(f"Failed to process match {match.id}: {e}")
    
    def season_reset(self, regression_factor: float = None) -> None:
        """
        Apply season reset: regress all ratings toward mean.
        
        This prevents ratings from drifting too far over time.
        """
        factor = regression_factor or self.SEASON_REGRESSION
        mean_rating = self.initial_rating
        
        for team_data in self.ratings.values():
            current = team_data.rating
            regressed = current + factor * (mean_rating - current)
            team_data.rating = regressed
        
        logger.info(f"Season reset applied with factor {factor}")
    
    def get_all_ratings(self) -> Dict[str, float]:
        """Get all team ratings as dict."""
        return {team: data.rating for team, data in self.ratings.items()}
    
    def get_top_teams(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N rated teams."""
        all_ratings = self.get_all_ratings()
        sorted_ratings = sorted(all_ratings.items(), key=lambda x: x[1], reverse=True)
        return sorted_ratings[:n]
    
    def predict_match(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float, float]:
        """
        Predict match probabilities based on ELO.
        
        Returns:
            (p_home, p_draw, p_away) - simplified ELO-based probabilities
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)
        
        # Apply home advantage
        p_home_win = self._expected_score(home_elo + self.home_advantage, away_elo)
        
        # Estimate draw probability (empirical: ~25% average in football)
        # Higher when teams are close in rating
        elo_diff = abs(home_elo - away_elo)
        draw_factor = max(0.15, 0.30 - elo_diff / 1000)
        
        # Adjust
        p_draw = draw_factor
        p_home = p_home_win * (1 - p_draw)
        p_away = (1 - p_home_win) * (1 - p_draw)
        
        # Normalize
        total = p_home + p_draw + p_away
        return p_home/total, p_draw/total, p_away/total


class EloBuilder:
    """
    Feature builder that uses ELO calculator.
    
    Returns ELO ratings for home and away teams at time of match.
    """
    
    def __init__(self, **elo_params):
        self.calculator = EloCalculator(**elo_params)
        self.is_fitted = False
    
    def fit(self, historical_matches: List[Match]) -> None:
        """Fit on historical data."""
        self.calculator.process_matches(historical_matches)
        self.is_fitted = True
    
    def get_features(
        self,
        home_team: str,
        away_team: str,
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get ELO features for a match."""
        home_elo = self.calculator.get_rating(home_team, as_of)
        away_elo = self.calculator.get_rating(away_team, as_of)
        
        # Get momentum if available
        home_record = self.calculator.get_elo_record(home_team)
        away_record = self.calculator.get_elo_record(away_team)
        
        return {
            "elo_home": home_elo,
            "elo_away": away_elo,
            "elo_diff": home_elo - away_elo,
            "elo_momentum_home": home_record.recent_momentum() if home_record else 0,
            "elo_momentum_away": away_record.recent_momentum() if away_record else 0,
        }
    
    def get_rating(self, team: str, as_of: Optional[datetime] = None) -> float:
        """Get single team's rating."""
        return self.calculator.get_rating(team, as_of)
