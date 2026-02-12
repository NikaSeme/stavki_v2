"""
Tests for data layer components.
"""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import os
from pydantic import ValidationError

from stavki.data.schemas import (
    Team, Match, OddsSnapshot, BestOdds, 
    League, Outcome, Prediction, ValueSignal,
    TeamFeatures, MatchFeatures,
)
from stavki.data.processors import (
    normalize_team_name, 
    OddsValidator, MatchValidator,
    LineMovementTracker, CLVTracker,
)
from stavki.data.storage import Database


class TestTeamNormalization:
    """Tests for team name normalization."""
    
    def test_basic_normalization(self):
        """Test basic string normalization."""
        assert normalize_team_name("Arsenal FC") == "arsenal"
        assert normalize_team_name("  Liverpool  ") == "liverpool"
        assert normalize_team_name("CHELSEA") == "chelsea"
    
    def test_known_aliases(self):
        """Test known team aliases are resolved."""
        assert normalize_team_name("Man Utd") == "manchester united"
        assert normalize_team_name("Man City") == "manchester city"
        assert normalize_team_name("Spurs") == "tottenham hotspur"
        assert normalize_team_name("Wolves") == "wolverhampton wanderers"
    
    def test_unicode_handling(self):
        """Test unicode character handling."""
        assert normalize_team_name("Bayern München") == "bayern munich"
        assert normalize_team_name("Atlético Madrid") == "atletico madrid"


class TestMatch:
    """Tests for Match schema."""
    
    def test_match_creation(self):
        """Test basic match creation."""
        match = Match(
            id="test_123",
            home_team=Team(name="Arsenal"),
            away_team=Team(name="Chelsea"),
            league=League.EPL,
            commence_time=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        
        assert match.id == "test_123"
        assert match.home_team.name == "Arsenal"
        assert not match.is_completed
        assert match.result is None
    
    def test_completed_match(self):
        """Test match with result."""
        match = Match(
            id="test_456",
            home_team=Team(name="Liverpool"),
            away_team=Team(name="Manchester United"),
            league=League.EPL,
            commence_time=datetime.now(timezone.utc) - timedelta(hours=24),
            home_score=2,
            away_score=1,
        )
        
        assert match.is_completed
        assert match.result == Outcome.HOME
        assert match.total_goals == 3
    
    def test_draw_result(self):
        """Test draw result detection."""
        match = Match(
            id="test_789",
            home_team=Team(name="Brighton"),
            away_team=Team(name="Fulham"),
            league=League.EPL,
            commence_time=datetime.now(timezone.utc),
            home_score=1,
            away_score=1,
        )
        
        assert match.result == Outcome.DRAW
    
    def test_hours_until_kickoff(self):
        """Test hours until kickoff calculation."""
        future_match = Match(
            id="future",
            home_team=Team(name="A"),
            away_team=Team(name="B"),
            league=League.EPL,
            commence_time=datetime.now(timezone.utc) + timedelta(hours=3),
        )
        
        hours = future_match.hours_until_kickoff()
        assert 2.9 < hours < 3.1


class TestOddsSnapshot:
    """Tests for odds handling."""
    
    def test_odds_creation(self):
        """Test odds snapshot creation."""
        odds = OddsSnapshot(
            match_id="test_match",
            bookmaker="bet365",
            timestamp=datetime.utcnow(),
            home_odds=2.5,
            draw_odds=3.2,
            away_odds=2.8,
        )
        
        assert odds.home_odds == 2.5
        assert odds.overround > 0  # Should have positive vig
    
    def test_no_vig_probs(self):
        """Test no-vig probability calculation."""
        odds = OddsSnapshot(
            match_id="test",
            bookmaker="pinnacle",
            timestamp=datetime.now(timezone.utc),
            home_odds=2.0,
            draw_odds=3.5,
            away_odds=4.0,
        )
        
        probs = odds.no_vig_probs()
        
        # Probs should sum to 1.0
        total = probs["home"] + probs["draw"] + probs["away"]
        assert abs(total - 1.0) < 0.01


class TestOddsValidator:
    """Tests for odds validation."""
    
    def test_valid_odds(self):
        """Test validation of good odds."""
        odds = OddsSnapshot(
            match_id="test",
            bookmaker="bet365",
            timestamp=datetime.now(timezone.utc),
            home_odds=2.5,
            draw_odds=3.2,
            away_odds=2.8,
        )
        
        result = OddsValidator.validate_snapshot(odds)
        assert result.is_valid
    
    def test_invalid_low_odds(self):
        """Test rejection of impossibly low odds at the schema level.
        
        Pydantic enforces gt=1.0 on odds fields, so invalid odds
        (below 1.0) are rejected at construction time.
        """
        with pytest.raises(ValidationError, match="greater_than"):
            OddsSnapshot(
                match_id="test",
                bookmaker="bad_book",
                timestamp=datetime.now(timezone.utc),
                home_odds=0.5,  # Invalid - below 1.0
                draw_odds=3.2,
                away_odds=2.8,
            )
    
    def test_outlier_detection(self):
        """Test outlier odds detection."""
        snapshots = [
            OddsSnapshot(
                match_id="test",
                bookmaker="book1",
                timestamp=datetime.now(timezone.utc),
                home_odds=2.5,
                away_odds=2.8,
            ),
            OddsSnapshot(
                match_id="test",
                bookmaker="book2",
                timestamp=datetime.now(timezone.utc),
                home_odds=2.4,
                away_odds=2.9,
            ),
            OddsSnapshot(
                match_id="test",
                bookmaker="outlier_book",
                timestamp=datetime.now(timezone.utc),
                home_odds=5.0,  # Way higher than others
                away_odds=1.5,
            ),
        ]
        
        outlier, gap = OddsValidator.find_outliers(snapshots, "home")
        assert outlier is not None
        assert outlier.bookmaker == "outlier_book"


class TestLineMovement:
    """Tests for line movement tracking."""
    
    def test_movement_detection(self):
        """Test basic line movement detection."""
        tracker = LineMovementTracker(match_id="test")
        
        # Add snapshots showing price drop
        tracker.add_snapshot(OddsSnapshot(
            match_id="test",
            bookmaker="book1",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            home_odds=2.5,
            away_odds=2.8,
        ))
        
        tracker.add_snapshot(OddsSnapshot(
            match_id="test",
            bookmaker="book1",
            timestamp=datetime.now(timezone.utc),
            home_odds=2.2,  # Dropped
            away_odds=3.0,
        ))
        
        movement = tracker.get_movement("home")
        assert movement is not None
        assert movement.total_movement_pct < 0  # Price dropped
    
    def test_steam_detection(self):
        """Test steam move detection."""
        tracker = LineMovementTracker(match_id="steam_test")
        
        # Big drop
        tracker.add_snapshot(OddsSnapshot(
            match_id="steam_test",
            bookmaker="book1",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=1),
            home_odds=3.0,
            away_odds=2.0,
        ))
        
        tracker.add_snapshot(OddsSnapshot(
            match_id="steam_test",
            bookmaker="book1",
            timestamp=datetime.now(timezone.utc),
            home_odds=2.5,  # 16% drop
            away_odds=2.3,
        ))
        
        steaming = tracker.detect_steam(threshold_pct=10)
        assert "home" in steaming


class TestCLVTracker:
    """Tests for CLV calculation."""
    
    def test_positive_clv(self):
        """Test positive CLV (got better odds)."""
        clv = CLVTracker.calculate_clv(bet_odds=2.5, closing_odds=2.3)
        assert clv > 0  # We beat the closing line
    
    def test_negative_clv(self):
        """Test negative CLV (got worse odds)."""
        clv = CLVTracker.calculate_clv(bet_odds=2.3, closing_odds=2.5)
        assert clv < 0  # We got worse than closing
    
    def test_clv_distribution(self):
        """Test CLV distribution analysis."""
        clvs = [5.0, 3.0, -2.0, 4.0, 1.0, -1.0, 2.0]
        
        analysis = CLVTracker.analyze_clv_distribution(clvs)
        
        assert analysis["mean"] > 0
        assert analysis["positive_rate"] > 0.5


class TestDatabase:
    """Tests for database operations."""
    
    @pytest.fixture
    def db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        
        db = Database(db_path)
        yield db
        
        os.unlink(db_path)
    
    def test_save_and_get_match(self, db):
        """Test saving and retrieving a match."""
        match = Match(
            id="db_test_1",
            home_team=Team(name="Arsenal"),
            away_team=Team(name="Chelsea"),
            league=League.EPL,
            commence_time=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        
        db.save_match(match)
        
        retrieved = db.get_match("db_test_1")
        assert retrieved is not None
        assert retrieved.id == "db_test_1"
        assert retrieved.league == League.EPL
    
    def test_update_result(self, db):
        """Test updating match result."""
        match = Match(
            id="result_test",
            home_team=Team(name="Liverpool"),
            away_team=Team(name="Manchester City"),
            league=League.EPL,
            commence_time=datetime.now(timezone.utc),
        )
        
        db.save_match(match)
        db.update_match_result("result_test", home_score=3, away_score=1)
        
        updated = db.get_match("result_test")
        assert updated.home_score == 3
        assert updated.away_score == 1


class TestTeamFeatures:
    """Tests for feature schemas."""
    
    def test_team_features(self):
        """Test team features computed properties."""
        features = TeamFeatures(
            team_name="Arsenal",
            computed_at=datetime.now(timezone.utc),
            elo_rating=1650,
            form_points=12,
            form_goals_scored=8,
            form_goals_conceded=3,
        )
        
        assert features.form_pct == 80.0  # 12/15 * 100
        assert features.goal_diff_form == 5
    
    def test_match_features_vector(self):
        """Test feature vector generation."""
        home = TeamFeatures(
            team_name="Home",
            computed_at=datetime.now(timezone.utc),
            elo_rating=1600,
            form_points=10,
        )
        away = TeamFeatures(
            team_name="Away",
            computed_at=datetime.now(timezone.utc),
            elo_rating=1500,
            form_points=8,
        )
        
        match_features = MatchFeatures(
            match_id="test",
            computed_at=datetime.now(timezone.utc),
            home_features=home,
            away_features=away,
            elo_diff=100,
            form_diff=2,
        )
        
        vector = match_features.to_feature_vector()
        assert len(vector) == len(MatchFeatures.feature_names())
        assert vector[0] == 1600  # home elo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
