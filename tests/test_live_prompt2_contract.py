"""
Prompt 2 Contract Tests for LivePredictor.

Tests enforce:
1) Missing critical features → invalid_for_bet=True, not recommended
2) invalid_for_bet fixtures excluded from get_recommendations output
3) Non-critical defaults allowed and substitution coverage logged
"""
import pytest
import logging
from unittest.mock import patch, MagicMock, PropertyMock
from datetime import datetime
import numpy as np
import pandas as pd

# Ensure project root on path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stavki.prediction.live import LivePrediction, CRITICAL_FEATURES_1X2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_prediction(**overrides) -> LivePrediction:
    """Create a LivePrediction with sensible defaults, overridable."""
    defaults = dict(
        fixture_id=1,
        home_team="TeamA",
        away_team="TeamB",
        league="epl",
        kickoff=datetime(2026, 3, 1, 15, 0),
        prob_home=0.4, prob_draw=0.3, prob_away=0.3,
        recommended=False,
        invalid_for_bet=False,
    )
    defaults.update(overrides)
    return LivePrediction(**defaults)


# ---------------------------------------------------------------------------
# Test 1: Missing critical features → invalid_for_bet=True
# ---------------------------------------------------------------------------
class TestCriticalFeatureGate:
    def test_missing_critical_features_marks_invalid(self):
        """If any critical feature is missing, prediction must be invalid_for_bet=True."""
        # Build a feature DataFrame MISSING 'elo_home' and 'elo_away'
        features = {
            'form_home_pts': [8.0],
            'form_away_pts': [6.0],
            'imp_home_norm': [0.4],
            'imp_draw_norm': [0.3],
            'imp_away_norm': [0.3],
            'elo_diff': [0.0],
            'form_diff': [2.0],
        }
        X = pd.DataFrame(features)

        # Import after path setup
        from stavki.prediction.live import LivePredictor

        # Patch __init__ to avoid Redis/API dependency
        with patch.object(LivePredictor, '__init__', lambda self, **kw: None):
            predictor = LivePredictor()
            predictor.model = None  # no model
            predictor.feature_cols = []
            predictor.min_ev = 0.03
            predictor.min_edge = 0.02
            predictor.model_alpha = 0.55
            predictor.kelly_fraction = 0.25
            predictor.LEAGUE_MAP = {8: 'epl'}

        # Patch _build_features to return our crafted DataFrame
        with patch.object(LivePredictor, '_build_features', return_value=(X, ['elo_home', 'elo_away'])):
            fixture = MagicMock()
            fixture.fixture_id = 100
            fixture.home_team = "TeamA"
            fixture.away_team = "TeamB"
            fixture.league_id = 8
            fixture.kickoff = datetime(2026, 3, 1, 15, 0)

            pred = predictor.predict_fixture(fixture, odds={'home': 2.0, 'draw': 3.3, 'away': 3.5})

        assert pred.invalid_for_bet is True, "Must be invalid when critical features missing"
        assert pred.recommended is False, "Must not be recommended when invalid"
        assert pred.best_bet is None, "best_bet must be None when invalid"
        assert pred.best_ev is None, "best_ev must be None when invalid"
        assert pred.stake_pct is None, "stake_pct must be None when invalid"
        assert 'elo_home' in pred.missing_critical_features
        assert 'elo_away' in pred.missing_critical_features

    def test_all_critical_present_not_invalid(self):
        """When all critical features present, invalid_for_bet must be False."""
        features = {col: [1.0] for col in CRITICAL_FEATURES_1X2}
        features.update({'elo_diff': [0.0], 'form_diff': [0.0]})
        X = pd.DataFrame(features)

        from stavki.prediction.live import LivePredictor

        with patch.object(LivePredictor, '__init__', lambda self, **kw: None):
            predictor = LivePredictor()
            predictor.model = None
            predictor.feature_cols = []
            predictor.min_ev = 0.03
            predictor.min_edge = 0.02
            predictor.model_alpha = 0.55
            predictor.kelly_fraction = 0.25
            predictor.LEAGUE_MAP = {8: 'epl'}

        with patch.object(LivePredictor, '_build_features', return_value=(X, [])):
            fixture = MagicMock()
            fixture.fixture_id = 200
            fixture.home_team = "TeamC"
            fixture.away_team = "TeamD"
            fixture.league_id = 8
            fixture.kickoff = datetime(2026, 3, 1, 15, 0)

            pred = predictor.predict_fixture(fixture, odds={'home': 2.0, 'draw': 3.3, 'away': 3.5})

        assert pred.invalid_for_bet is False, "All critical features present → must be valid"


# ---------------------------------------------------------------------------
# Test 2: get_recommendations excludes invalid_for_bet
# ---------------------------------------------------------------------------
class TestRecommendationsExcludeInvalid:
    def test_invalid_fixtures_excluded_from_recommendations(self):
        """get_recommendations must never return fixtures with invalid_for_bet=True."""
        from stavki.prediction.live import LivePredictor

        preds = [
            _make_prediction(fixture_id=1, recommended=True, invalid_for_bet=False,
                             best_ev=0.05, best_bet="Home"),
            _make_prediction(fixture_id=2, recommended=True, invalid_for_bet=True,
                             best_ev=0.08, best_bet="Away"),
            _make_prediction(fixture_id=3, recommended=True, invalid_for_bet=False,
                             best_ev=0.04, best_bet="Draw"),
        ]

        with patch.object(LivePredictor, '__init__', lambda self, **kw: None):
            predictor = LivePredictor()

        with patch.object(LivePredictor, 'predict_upcoming', return_value=preds):
            recs = predictor.get_recommendations(days=7, max_bets=10)

        rec_ids = [r.fixture_id for r in recs]
        assert 2 not in rec_ids, "invalid_for_bet fixture must be excluded from recommendations"
        assert 1 in rec_ids
        assert 3 in rec_ids


# ---------------------------------------------------------------------------
# Test 3: Non-critical defaults allowed and coverage logged
# ---------------------------------------------------------------------------
class TestNonCriticalDefaults:
    def test_non_critical_defaults_allowed_coverage_logged(self, caplog):
        """Non-critical features may use defaults; substitution_coverage_pct must be set."""
        # All critical features present, but some non-critical missing
        features = {col: [1.0] for col in CRITICAL_FEATURES_1X2}
        features.update({'elo_diff': [0.0], 'form_diff': [0.0]})
        X = pd.DataFrame(features)

        from stavki.prediction.live import LivePredictor

        with patch.object(LivePredictor, '__init__', lambda self, **kw: None):
            predictor = LivePredictor()
            predictor.model = None
            predictor.feature_cols = list(CRITICAL_FEATURES_1X2) + ['rolling_fouls_home', 'xg_home']
            predictor.min_ev = 0.03
            predictor.min_edge = 0.02
            predictor.model_alpha = 0.55
            predictor.kelly_fraction = 0.25
            predictor.LEAGUE_MAP = {8: 'epl'}

        with patch.object(LivePredictor, '_build_features', return_value=(X, [])):
            fixture = MagicMock()
            fixture.fixture_id = 300
            fixture.home_team = "TeamE"
            fixture.away_team = "TeamF"
            fixture.league_id = 8
            fixture.kickoff = datetime(2026, 3, 1, 15, 0)

            with caplog.at_level(logging.INFO, logger="stavki.prediction.live"):
                pred = predictor.predict_fixture(fixture, odds={'home': 2.0, 'draw': 3.3, 'away': 3.5})

        assert pred.invalid_for_bet is False, "Non-critical defaults should not invalidate prediction"
        assert pred.substitution_coverage_pct is not None, "substitution_coverage_pct must be set"


# ---------------------------------------------------------------------------
# Test 4: Model-column gate — missing_cols branch catches critical columns
# ---------------------------------------------------------------------------
class TestModelColumnCriticalGate:
    def test_model_column_critical_missing_returns_invalid(self):
        """If model.feature_cols includes a critical feature not in the X DataFrame,
        predict_fixture must return invalid_for_bet=True even though _build_features
        returned no missing_critical (i.e. the second gate catches it)."""
        from stavki.prediction.live import LivePredictor

        # Build X with ALL critical features present (first gate passes)
        features = {col: [1.0] for col in CRITICAL_FEATURES_1X2}
        features.update({'elo_diff': [0.0], 'form_diff': [0.0]})
        X = pd.DataFrame(features)

        # But model expects 'elo_away' which we REMOVE from X to simulate gap
        X = X.drop(columns=['elo_away'])

        mock_model = MagicMock()
        mock_model.predict = MagicMock(return_value=[])

        with patch.object(LivePredictor, '__init__', lambda self, **kw: None):
            predictor = LivePredictor()
            predictor.model = mock_model
            # feature_cols includes elo_away which is NOT in X
            predictor.feature_cols = list(CRITICAL_FEATURES_1X2) + ['elo_diff', 'form_diff']
            predictor.min_ev = 0.03
            predictor.min_edge = 0.02
            predictor.model_alpha = 0.55
            predictor.kelly_fraction = 0.25
            predictor.LEAGUE_MAP = {8: 'epl'}

        # _build_features returns X without elo_away, but missing_critical=[]
        # (simulates: _build_features itself didn't detect a gap, but model needs elo_away)
        with patch.object(LivePredictor, '_build_features', return_value=(X, [])):
            fixture = MagicMock()
            fixture.fixture_id = 400
            fixture.home_team = "TeamG"
            fixture.away_team = "TeamH"
            fixture.league_id = 8
            fixture.kickoff = datetime(2026, 3, 1, 15, 0)

            pred = predictor.predict_fixture(fixture, odds={'home': 2.0, 'draw': 3.3, 'away': 3.5})

        assert pred.invalid_for_bet is True, "Model-column gate must catch critical missing"
        assert pred.recommended is False
        assert pred.best_bet is None
        assert pred.best_ev is None
        assert pred.stake_pct is None
        assert 'elo_away' in pred.missing_critical_features
        assert "Model requires" in pred.invalid_reason

