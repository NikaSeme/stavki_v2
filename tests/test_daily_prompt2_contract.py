"""
Prompt 2 Contract Tests for DailyPipeline.

Tests enforce:
1) _map_features_to_model_inputs marks rows invalid when critical features missing
2) Recommendation path excludes invalid_for_bet rows
3) Non-critical substitution coverage appears in logs
"""
import pytest
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

# Ensure project root on path
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stavki.pipelines.daily import DailyPipeline, CRITICAL_FEATURES_1X2 as DAILY_CRITICAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_feature_columns_json(tmpdir: Path, columns: list) -> Path:
    """Write a temporary feature_columns.json."""
    p = tmpdir / "models" / "feature_columns.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(columns, f)
    return p


# ---------------------------------------------------------------------------
# Test 1: _map_features_to_model_inputs marks invalid when critical missing
# ---------------------------------------------------------------------------
class TestMapFeaturesInvalidGate:
    def test_marks_invalid_when_critical_missing(self, tmp_path):
        """When critical features are not in the DataFrame and must be filled,
        rows must get _invalid_for_bet=True."""
        # Create feature_columns.json that includes critical + non-critical columns
        all_cols = list(DAILY_CRITICAL) + ["rolling_fouls_home", "xg_diff"]
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        fc_path = models_dir / "feature_columns.json"
        with open(fc_path, "w") as f:
            json.dump(all_cols, f)

        # DataFrame MISSING critical columns (elo_home, elo_away, etc.)
        df = pd.DataFrame({
            "match_id": ["m1", "m2"],
            "home_team": ["TeamA", "TeamC"],
            "away_team": ["TeamB", "TeamD"],
            # Only non-critical columns present
            "rolling_fouls_home": [11.0, 13.0],
            "xg_diff": [0.2, -0.1],
        })

        with patch.object(DailyPipeline, '__init__', lambda self, **kw: None):
            pipeline = DailyPipeline()

        # Monkeypatch __file__ inside the daily module so Path(__file__) resolves to tmp_path
        import stavki.pipelines.daily as daily_mod
        # Fake __file__ path: tmp_path/stavki/pipelines/daily.py
        # so that Path(__file__).parent.parent.parent == tmp_path
        fake_file = str(tmp_path / "stavki" / "pipelines" / "daily.py")
        (tmp_path / "stavki" / "pipelines").mkdir(parents=True, exist_ok=True)
        original_file = daily_mod.__file__

        try:
            daily_mod.__file__ = fake_file
            result = pipeline._map_features_to_model_inputs(df)
        finally:
            daily_mod.__file__ = original_file

        # Verify _invalid_for_bet column exists and is True for all rows
        assert "_invalid_for_bet" in result.columns, \
            "_invalid_for_bet column must exist when critical features are missing"
        assert result["_invalid_for_bet"].all(), \
            "All rows must be marked _invalid_for_bet=True when elo/form columns were missing"

    def test_valid_when_all_critical_present(self, tmp_path):
        """When all critical features are present, rows must NOT be invalid."""
        all_cols = list(DAILY_CRITICAL) + ["rolling_fouls_home"]
        _make_feature_columns_json(tmp_path, all_cols)

        # DataFrame WITH all critical columns
        data = {col: [1.0, 2.0] for col in DAILY_CRITICAL}
        data.update({
            "match_id": ["m1", "m2"],
            "home_team": ["TeamA", "TeamC"],
            "away_team": ["TeamB", "TeamD"],
        })
        df = pd.DataFrame(data)

        with patch.object(DailyPipeline, '__init__', lambda self, **kw: None):
            pipeline = DailyPipeline()

        result = pipeline._map_features_to_model_inputs(df)

        if "_invalid_for_bet" in result.columns:
            assert not result["_invalid_for_bet"].any(), \
                "No rows should be invalid when all critical features present"


# ---------------------------------------------------------------------------
# Test 2: Recommendation path excludes invalid_for_bet rows
# ---------------------------------------------------------------------------
class TestRecommendationExcludesInvalid:
    def test_find_value_bets_skips_invalid_rows(self):
        """_find_value_bets must not produce BetCandidates for invalid_for_bet rows."""
        # We test indirectly: features_df with _invalid_for_bet=True should
        # produce zero candidates for that match_id
        from stavki.pipelines.daily import DailyPipeline

        with patch.object(DailyPipeline, '__init__', lambda self, **kw: None):
            pipeline = DailyPipeline()
            pipeline.config = MagicMock()
            pipeline.config.min_ev = 0.03
            pipeline.config.min_confidence = 0.05
            pipeline.config.max_divergence = 0.25
            pipeline._blender = MagicMock()
            pipeline._blender.blend = MagicMock(return_value=0.55)
            pipeline.OUTCOME_TO_MARKET = {
                "home": "1x2", "draw": "1x2", "away": "1x2",
            }

        matches_df = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "league": ["epl", "epl"],
            "_invalid_for_bet": [True, False],
        })

        model_probs = {
            "e1": {"1x2": {"home": 0.5, "draw": 0.25, "away": 0.25}},
            "e2": {"1x2": {"home": 0.5, "draw": 0.25, "away": 0.25}},
        }
        market_probs = {
            "e1": {"home": 0.4, "draw": 0.3, "away": 0.3},
            "e2": {"home": 0.4, "draw": 0.3, "away": 0.3},
        }
        best_prices = pd.DataFrame({
            "event_id": ["e1", "e1", "e2", "e2"],
            "outcome_name": ["home", "away", "home", "away"],
            "outcome_price": [2.2, 3.5, 2.2, 3.5],
            "bookmaker_key": ["bk1", "bk1", "bk1", "bk1"],
        })

        candidates = pipeline._find_value_bets(matches_df, model_probs, market_probs, best_prices)

        invalid_match_ids = [c.match_id for c in candidates if c.match_id == "e1"]
        assert len(invalid_match_ids) == 0, \
            "No BetCandidates should be produced for invalid_for_bet=True matches"


# ---------------------------------------------------------------------------
# Test 3: Non-critical substitution coverage in logs
# ---------------------------------------------------------------------------
class TestSubstitutionCoverageLogged:
    def test_coverage_logged(self, caplog, tmp_path):
        """_map_features_to_model_inputs must log non-critical substitution coverage %."""
        all_cols = list(DAILY_CRITICAL) + ["rolling_fouls_home", "xg_diff", "rolling_corners_home"]
        _make_feature_columns_json(tmp_path, all_cols)

        # All critical present, non-critical missing
        data = {col: [1.0, 2.0] for col in DAILY_CRITICAL}
        data.update({
            "match_id": ["m1", "m2"],
            "home_team": ["TeamA", "TeamC"],
            "away_team": ["TeamB", "TeamD"],
        })
        df = pd.DataFrame(data)

        with patch.object(DailyPipeline, '__init__', lambda self, **kw: None):
            pipeline = DailyPipeline()

        with caplog.at_level(logging.INFO, logger="stavki.pipelines.daily"):
            result = pipeline._map_features_to_model_inputs(df)

        coverage_logged = any("substitution coverage" in r.message.lower() for r in caplog.records)
        assert coverage_logged, "Must log non-critical substitution coverage percentage"


# ---------------------------------------------------------------------------
# Test 4: run() path propagates invalid flag from features_df to matches_df
# ---------------------------------------------------------------------------
class TestRunPathPropagatesInvalidFlag:
    def test_run_path_merges_invalid_flag(self):
        """The run() method must merge _invalid_for_bet from features_df into
        matches_df, so _find_value_bets can exclude invalid rows.

        Strategy: patch _build_features to return features_df with _invalid_for_bet;
        verify matches_df passed to _find_value_bets contains the flag."""
        from stavki.pipelines.daily import DailyPipeline, PipelineConfig

        # Tracks what _find_value_bets receives
        captured_matches_df = {}

        def mock_find_value_bets(self, matches_df, model_probs, market_probs, best_prices):
            captured_matches_df['df'] = matches_df.copy()
            return []  # no candidates

        with patch.object(DailyPipeline, '__init__', lambda self, **kw: None):
            pipeline = DailyPipeline()
            pipeline.config = MagicMock()
            pipeline.config.save_predictions = False
            pipeline._blender = MagicMock()

        # Create matches_df WITHOUT _invalid_for_bet
        matches_df = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "home_team": ["TeamA", "TeamC"],
            "away_team": ["TeamB", "TeamD"],
            "league": ["epl", "epl"],
        })

        # Create features_df WITH _invalid_for_bet (as would come from _build_features)
        features_df = pd.DataFrame({
            "event_id": ["e1", "e2"],
            "_invalid_for_bet": [True, False],
            "elo_home": [np.nan, 1600.0],
            "elo_away": [np.nan, 1400.0],
        })

        model_probs = {
            "e1": {"1x2": {"home": 0.5}},
            "e2": {"1x2": {"home": 0.5}},
        }

        with patch.object(DailyPipeline, '_init_components'):
            with patch.object(DailyPipeline, '_build_features', return_value=features_df):
                with patch.object(DailyPipeline, '_get_predictions', return_value=model_probs):
                    with patch.object(DailyPipeline, '_select_best_prices', return_value=pd.DataFrame()):
                        with patch.object(DailyPipeline, '_compute_market_probs', return_value={}):
                            with patch.object(DailyPipeline, '_find_value_bets', mock_find_value_bets):
                                with patch.object(DailyPipeline, '_apply_filters', return_value=[]):
                                    with patch.object(DailyPipeline, '_calculate_stakes', return_value=[]):
                                        pipeline.run(
                                            odds_df=pd.DataFrame({"event_id": ["e1"]}),
                                            matches_df=matches_df,
                                        )

        # Verify that _find_value_bets received matches_df WITH _invalid_for_bet
        assert 'df' in captured_matches_df, "_find_value_bets must have been called"
        received_df = captured_matches_df['df']
        assert "_invalid_for_bet" in received_df.columns, \
            "matches_df passed to _find_value_bets must contain _invalid_for_bet from features_df"
        
        # e1 should be True (invalid), e2 should be False (valid)
        e1_row = received_df[received_df["event_id"] == "e1"]
        e2_row = received_df[received_df["event_id"] == "e2"]
        
        assert len(e1_row) == 1
        assert e1_row.iloc[0]["_invalid_for_bet"] == True, \
            "e1 must be marked invalid_for_bet=True"
        assert len(e2_row) == 1
        assert e2_row.iloc[0]["_invalid_for_bet"] == False, \
            "e2 must be marked invalid_for_bet=False"

