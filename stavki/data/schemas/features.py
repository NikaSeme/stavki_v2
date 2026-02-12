"""
Feature schemas for the feature engineering pipeline.

These define the structure of computed features
used by the ML models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, computed_field


class TeamFeatures(BaseModel):
    """
    Computed features for a single team.
    
    Generated from historical match data.
    """
    
    team_name: str
    computed_at: datetime
    
    # ELO Rating
    elo_rating: float = 1500.0
    elo_change_last_5: float = 0.0  # Momentum
    
    # Form (last 5 games)
    form_points: int = 0  # 0-15 (3 per win, 1 per draw)
    form_wins: int = 0
    form_draws: int = 0
    form_losses: int = 0
    form_goals_scored: int = 0
    form_goals_conceded: int = 0
    
    # Goals (last 10 games for stability)
    avg_goals_scored: float = 0.0
    avg_goals_conceded: float = 0.0
    
    # Home/Away specific (last 5 at home/away)
    home_form_points: int = 0
    away_form_points: int = 0
    home_goals_scored_avg: float = 0.0
    away_goals_scored_avg: float = 0.0
    
    # Streaks
    win_streak: int = 0
    unbeaten_streak: int = 0
    clean_sheets_streak: int = 0
    
    # Season totals
    season_wins: int = 0
    season_draws: int = 0
    season_losses: int = 0
    season_goal_diff: int = 0
    season_position: Optional[int] = None
    
    @computed_field
    @property
    def form_pct(self) -> float:
        """Form as percentage of maximum."""
        return self.form_points / 15 * 100
    
    @computed_field
    @property
    def goal_diff_form(self) -> int:
        """Goal difference in form window."""
        return self.form_goals_scored - self.form_goals_conceded
    
    # Average goals per team per game across top-5 European leagues (2015-2025)
    # Source: computed from historical data — EPL, La Liga, Serie A, Bundesliga, Ligue 1
    # This is used as the baseline for attack/defense strength normalization.
    # Override per-league via LeagueStats when available.
    LEAGUE_AVG_GOALS: float = 1.37

    @computed_field
    @property
    def attack_strength(self) -> float:
        """Relative attack strength (goals scored / league avg)."""
        if self.avg_goals_scored > 0:
            return self.avg_goals_scored / self.LEAGUE_AVG_GOALS
        return 1.0
    
    @computed_field
    @property
    def defense_strength(self) -> float:
        """Relative defense strength (inverted - lower is better)."""
        if self.avg_goals_conceded > 0:
            return self.avg_goals_conceded / self.LEAGUE_AVG_GOALS
        return 1.0


class H2HFeatures(BaseModel):
    """
    Head-to-head features between two teams.
    """
    
    home_team: str
    away_team: str
    computed_at: datetime
    
    # Total H2H record
    total_matches: int = 0
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    
    # Recent H2H (last 5 meetings)
    recent_home_wins: int = 0
    recent_draws: int = 0
    recent_away_wins: int = 0
    
    # Goal stats
    avg_total_goals: float = 0.0
    avg_home_goals: float = 0.0
    avg_away_goals: float = 0.0
    
    # Trends
    home_winning_trend: bool = False  # Home team improving in H2H
    away_winning_trend: bool = False
    
    @computed_field
    @property
    def home_win_rate(self) -> float:
        if self.total_matches == 0:
            return 0.5
        return self.home_wins / self.total_matches
    
    @computed_field
    @property
    def draw_rate(self) -> float:
        if self.total_matches == 0:
            return 0.25
        return self.draws / self.total_matches


class MatchFeatures(BaseModel):
    """
    Complete feature vector for a match prediction.
    
    This is what gets fed to the ML models.
    """
    
    match_id: str
    computed_at: datetime
    
    # Team features
    home_features: TeamFeatures
    away_features: TeamFeatures
    h2h_features: Optional[H2HFeatures] = None
    
    # Derived comparative features
    elo_diff: float = 0.0  # home - away
    form_diff: float = 0.0
    attack_diff: float = 0.0
    defense_diff: float = 0.0
    
    # Context features (for temporal model)
    is_weekend: bool = False
    is_evening: bool = False
    is_prime_time: bool = False
    day_of_week: int = 0  # 0 = Monday
    month: int = 1
    
    # Match importance (if available)
    is_derby: bool = False
    is_title_decider: bool = False
    is_relegation: bool = False
    
    # Market features (from odds)
    implied_home_prob: Optional[float] = None
    implied_draw_prob: Optional[float] = None
    implied_away_prob: Optional[float] = None
    market_confidence: Optional[float] = None  # 1 - overround
    
    def to_feature_vector(self) -> List[float]:
        """Convert to flat feature vector for models."""
        return [
            # ELO
            self.home_features.elo_rating,
            self.away_features.elo_rating,
            self.elo_diff,
            
            # Form
            self.home_features.form_points,
            self.away_features.form_points,
            self.form_diff,
            
            # Goals
            self.home_features.avg_goals_scored,
            self.home_features.avg_goals_conceded,
            self.away_features.avg_goals_scored,
            self.away_features.avg_goals_conceded,
            
            # Home/Away advantage
            self.home_features.home_form_points,
            self.away_features.away_form_points,
            
            # Streaks
            self.home_features.win_streak,
            self.away_features.win_streak,
            
            # Attack/Defense
            self.home_features.attack_strength,
            self.home_features.defense_strength,
            self.away_features.attack_strength,
            self.away_features.defense_strength,
            
            # Context
            float(self.is_weekend),
            float(self.is_evening),
            float(self.day_of_week),
            
            # H2H (if available)
            self.h2h_features.home_win_rate if self.h2h_features else 0.5,
            self.h2h_features.avg_total_goals if self.h2h_features else 2.5,
            
            # Market context
            # Uniform prior for 3-way market (1/3 for each outcome)
            self.implied_home_prob or 1/3,
            self.implied_draw_prob or 1/3,
            self.implied_away_prob or 1/3,
        ]
    
    @classmethod
    def feature_names(cls) -> List[str]:
        """Get feature names in same order as to_feature_vector."""
        return [
            "elo_home", "elo_away", "elo_diff",
            "form_home", "form_away", "form_diff",
            "goals_scored_home", "goals_conceded_home",
            "goals_scored_away", "goals_conceded_away",
            "home_form_at_home", "away_form_at_away",
            "win_streak_home", "win_streak_away",
            "attack_home", "defense_home",
            "attack_away", "defense_away",
            "is_weekend", "is_evening", "day_of_week",
            "h2h_home_win_rate", "h2h_avg_goals",
            "market_home_prob", "market_draw_prob", "market_away_prob",
        ]


class LeagueStats(BaseModel):
    """
    League-level statistics for normalization.
    """
    
    league: str
    season: str
    computed_at: datetime
    
    # Scoring
    avg_goals_per_game: float = 2.7
    avg_home_goals: float = 1.5
    avg_away_goals: float = 1.2
    
    # Results distribution
    home_win_rate: float = 0.45
    draw_rate: float = 0.25
    away_win_rate: float = 0.30
    
    # Other markets
    btts_rate: float = 0.50  # Both teams to score
    over_2_5_rate: float = 0.55
    
    # For attack/defense strength normalization
    total_matches: int = 0
