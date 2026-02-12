"""
Tests for features layer.
"""

import pytest
from datetime import datetime, timedelta, timezone

from stavki.data.schemas import Match, Team, League, Outcome
from stavki.features import (
    EloCalculator, EloBuilder,
    FormCalculator, FormBuilder,
    H2HBuilder,
    calculate_disagreement, detect_contrarian_opportunity,
    FeatureRegistry,
)


def create_match(
    home: str,
    away: str, 
    home_score: int,
    away_score: int,
    days_ago: int = 0
) -> Match:
    """Helper to create test matches."""
    return Match(
        id=f"{home}_{away}_{days_ago}",
        home_team=Team(name=home, normalized_name=home.lower()),
        away_team=Team(name=away, normalized_name=away.lower()),
        league=League.EPL,
        commence_time=datetime.now(timezone.utc) - timedelta(days=days_ago),
        home_score=home_score,
        away_score=away_score,
    )


class TestEloCalculator:
    """Tests for ELO calculator."""
    
    def test_initial_rating(self):
        """New teams start at 1500."""
        calc = EloCalculator()
        assert calc.get_rating("new_team") == 1500.0
    
    def test_rating_changes_after_match(self):
        """Ratings change after processing a match."""
        calc = EloCalculator()
        
        match = create_match("team_a", "team_b", home_score=2, away_score=0, days_ago=1)
        calc.process_match(match)
        
        # Winner should gain, loser should lose
        assert calc.get_rating("team_a") > 1500
        assert calc.get_rating("team_b") < 1500
    
    def test_draw_small_change(self):
        """Draw results in small rating changes.
        
        When team_a is stronger AND plays at home, it's expected to win.
        A draw is an underperformance, so its rating should decrease.
        """
        calc = EloCalculator()
        
        # Make team_a stronger first
        match1 = create_match("team_a", "weak", 5, 0, days_ago=10)
        calc.process_match(match1)
        strong_rating = calc.get_rating("team_a")
        
        # team_a at HOME draws with equal team — disappointing for the favorite
        match2 = create_match("team_a", "equal", 1, 1, days_ago=5)
        calc.process_match(match2)
        
        # The stronger HOME team should lose slight rating on draw
        assert calc.get_rating("team_a") < strong_rating
    
    def test_goal_diff_matters(self):
        """Bigger wins should result in bigger rating changes."""
        calc1 = EloCalculator()
        calc2 = EloCalculator()
        
        # 1-0 win
        match1 = create_match("team_a", "team_b", 1, 0, days_ago=1)
        calc1.process_match(match1)
        change_small = calc1.get_rating("team_a") - 1500
        
        # 5-0 win
        match2 = create_match("team_a", "team_b", 5, 0, days_ago=1)
        calc2.process_match(match2)
        change_big = calc2.get_rating("team_a") - 1500
        
        assert change_big > change_small
    
    def test_chronological_processing(self):
        """Matches must be processed in order."""
        calc = EloCalculator()
        
        matches = [
            create_match("team_a", "team_b", 2, 0, days_ago=10),
            create_match("team_a", "team_b", 0, 2, days_ago=5),
            create_match("team_a", "team_b", 1, 1, days_ago=1),
        ]
        
        calc.process_matches(matches)
        
        # All matches processed
        assert calc.ratings["team_a"].matches_played == 3
        assert calc.ratings["team_b"].matches_played == 3


class TestFormCalculator:
    """Tests for form calculator."""
    
    def test_form_points(self):
        """Win=3, Draw=1, Loss=0."""
        calc = FormCalculator(window=5)
        
        matches = [
            create_match("team_a", "x", 2, 0, days_ago=5),  # Win
            create_match("team_a", "y", 1, 1, days_ago=4),  # Draw
            create_match("team_a", "z", 0, 1, days_ago=3),  # Loss
        ]
        
        stats = calc.calculate("team_a", matches)
        
        assert stats.wins == 1
        assert stats.draws == 1
        assert stats.losses == 1
        assert stats.points == 4  # 3 + 1 + 0
    
    def test_respects_window(self):
        """Only last N matches counted."""
        calc = FormCalculator(window=3)
        
        matches = [
            create_match("team_a", "x", 2, 0, days_ago=10),  # Too old
            create_match("team_a", "y", 2, 0, days_ago=3),
            create_match("team_a", "z", 2, 0, days_ago=2),
            create_match("team_a", "w", 2, 0, days_ago=1),
        ]
        
        stats = calc.calculate("team_a", matches)
        
        assert stats.matches == 3  # Only last 3
    
    def test_streak_detection(self):
        """Detect winning streaks."""
        calc = FormCalculator(window=10)
        
        matches = [
            create_match("team_a", "x", 2, 0, days_ago=5),
            create_match("team_a", "y", 3, 1, days_ago=4),
            create_match("team_a", "z", 1, 0, days_ago=3),
            create_match("team_a", "w", 0, 1, days_ago=2),  # Loss breaks streak
            create_match("team_a", "v", 2, 1, days_ago=1),  # Win
        ]
        
        streak = calc.get_streak("team_a", matches, streak_type="win")
        
        assert streak == 1  # Only 1 win after the loss


class TestH2HBuilder:
    """Tests for H2H feature builder."""
    
    def test_h2h_record(self):
        """Track head-to-head record."""
        builder = H2HBuilder(max_meetings=10)
        
        matches = [
            create_match("team_a", "team_b", 2, 0, days_ago=100),  # A wins
            create_match("team_b", "team_a", 1, 1, days_ago=50),   # Draw
            create_match("team_a", "team_b", 1, 2, days_ago=10),   # B wins
        ]
        
        h2h = builder.calculate("team_a", "team_b", matches)
        
        assert h2h.total_matches == 3
        assert h2h.home_wins == 1  # A's perspective
        assert h2h.draws == 1
        assert h2h.away_wins == 1
    
    def test_no_h2h_history(self):
        """Handle teams that never played."""
        builder = H2HBuilder()
        
        matches = [
            create_match("team_x", "team_y", 2, 0, days_ago=10),  # Different teams
        ]
        
        h2h = builder.calculate("team_a", "team_b", matches)
        
        assert h2h.total_matches == 0
        assert h2h.home_win_rate == 0.5  # Default


class TestDisagreement:
    """Tests for disagreement signal."""
    
    def test_no_disagreement(self):
        """Models that agree should have low disagreement."""
        result = calculate_disagreement(
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
        )
        
        assert result["disagreement_mean"] < 0.01
    
    def test_high_disagreement(self):
        """Models that disagree should have high disagreement."""
        result = calculate_disagreement(
            [0.7, 0.2, 0.1],  # Poisson: home win
            [0.3, 0.3, 0.4],  # CatBoost: away win
            [0.2, 0.5, 0.3],  # Neural: draw
        )
        
        assert result["disagreement_mean"] > 0.10
    
    def test_contrarian_detection(self):
        """Detect when models agree but differ from market."""
        model_probs = {
            "poisson": [0.6, 0.25, 0.15],
            "catboost": [0.58, 0.27, 0.15],
            "neural": [0.62, 0.23, 0.15],
        }
        market_probs = [0.40, 0.30, 0.30]  # Market favors away
        
        opp = detect_contrarian_opportunity(model_probs, market_probs, threshold=0.10)
        
        assert opp is not None
        assert opp["outcome"] == "home"
        assert opp["type"] == "model_favors"


class TestFeatureRegistry:
    """Tests for feature registry."""
    
    def test_fit_and_compute(self):
        """Can fit and compute features."""
        matches = [
            create_match("team_a", "team_b", 2, 0, days_ago=50),
            create_match("team_a", "team_c", 1, 1, days_ago=40),
            create_match("team_b", "team_c", 0, 1, days_ago=30),
            create_match("team_a", "team_b", 3, 1, days_ago=20),
            create_match("team_c", "team_a", 0, 2, days_ago=10),
        ]
        
        registry = FeatureRegistry()
        registry.fit(matches)
        
        features = registry.compute("team_a", "team_b")
        
        # Should have ELO features
        assert "elo_home" in features
        assert "elo_away" in features
        assert "elo_diff" in features
        
        # Should have form features
        assert "form_home" in features
        assert "form_away" in features
        
        # Should have H2H features
        assert "h2h_matches" in features
    
    def test_elo_reflects_performance(self):
        """Better teams should have higher ELO."""
        matches = [
            create_match("winner", "loser", 3, 0, days_ago=10),
            create_match("loser", "winner", 0, 2, days_ago=5),
        ]
        
        registry = FeatureRegistry()
        registry.fit(matches)
        
        winner_elo = registry.get_elo_rating("winner")
        loser_elo = registry.get_elo_rating("loser")
        
        assert winner_elo > loser_elo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
