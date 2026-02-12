"""
Tests for strategy.filters — BetFilters and MetaFilter
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from datetime import datetime, timedelta

from stavki.strategy.filters import BetFilters, MetaFilter, FilterResult
from stavki.strategy.ev import EVResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ev(ev_val=0.05, prob=0.55, odds=2.5, edge=0.05, match_id="m1") -> EVResult:
    return EVResult(
        match_id=match_id,
        market="1X2",
        selection="home",
        model_prob=prob,
        odds=odds,
        ev=ev_val,
        edge_pct=edge,
        implied_prob=1.0 / odds,
    )


# ---------------------------------------------------------------------------
# BetFilters tests
# ---------------------------------------------------------------------------

class TestBetFilters:
    """Test the BetFilters guardrails."""

    def test_good_bet_passes(self):
        filters = BetFilters()
        ev = _ev(ev_val=0.10, prob=0.55, odds=2.5, edge=0.05)
        result = filters.apply_all_filters(ev, league="EPL")
        assert result.passed is True

    def test_low_ev_rejected(self):
        filters = BetFilters(config={"min_ev": 0.05})
        ev = _ev(ev_val=0.01)
        result = filters.apply_all_filters(ev)
        assert result.passed is False
        assert "EV too low" in result.reason

    def test_high_ev_rejected(self):
        filters = BetFilters(config={"max_ev": 0.50})
        ev = _ev(ev_val=0.60)
        result = filters.apply_all_filters(ev)
        assert result.passed is False
        assert "suspiciously high" in result.reason

    def test_prob_outside_range_rejected(self):
        filters = BetFilters(config={"min_prob": 0.10, "max_prob": 0.90})
        ev = _ev(prob=0.05)
        result = filters.apply_all_filters(ev)
        assert result.passed is False
        assert "Prob outside range" in result.reason

    def test_odds_outside_range_rejected(self):
        filters = BetFilters(config={"min_odds": 1.20, "max_odds": 10.0})
        ev = _ev(odds=12.0)
        result = filters.apply_all_filters(ev)
        assert result.passed is False
        assert "Odds outside range" in result.reason

    def test_low_edge_rejected(self):
        filters = BetFilters(config={"min_edge": 0.05})
        ev = _ev(edge=0.01)
        result = filters.apply_all_filters(ev)
        assert result.passed is False
        assert "Edge too low" in result.reason

    def test_excluded_league_rejected(self):
        filters = BetFilters(config={"excluded_leagues": {"SerieA"}})
        ev = _ev()
        result = filters.apply_all_filters(ev, league="SerieA")
        assert result.passed is False
        assert "Excluded league" in result.reason

    def test_too_close_to_kickoff_rejected(self):
        filters = BetFilters(config={"min_hours_before_match": 2})
        ev = _ev()
        kickoff = datetime.now() + timedelta(hours=0.5)
        result = filters.apply_all_filters(ev, match_datetime=kickoff)
        assert result.passed is False
        assert "Too close" in result.reason

    def test_odds_drift_rejected(self):
        filters = BetFilters(config={"max_odds_drift": 0.10})
        ev = _ev(odds=2.5)
        result = filters.apply_all_filters(ev, opening_odds=2.0)
        assert result.passed is False
        assert "drifted" in result.reason


# ---------------------------------------------------------------------------
# MetaFilter tests
# ---------------------------------------------------------------------------

class TestMetaFilter:
    """Test multi-model agreement filter."""

    def test_all_agree_passes(self):
        mf = MetaFilter(min_models_agree=2, max_disagreement=0.15)
        preds = {
            "model_a": {"home": 0.60, "draw": 0.25, "away": 0.15},
            "model_b": {"home": 0.55, "draw": 0.25, "away": 0.20},
            "model_c": {"home": 0.58, "draw": 0.22, "away": 0.20},
        }
        result = mf.check_agreement(preds, "home")
        assert result.passed is True
        assert result.details["agree_count"] == 3

    def test_not_enough_models_fails(self):
        mf = MetaFilter(min_models_agree=3)
        preds = {
            "model_a": {"home": 0.60, "draw": 0.25, "away": 0.15},
        }
        result = mf.check_agreement(preds, "home")
        assert result.passed is False
        assert "Not enough models" in result.reason

    def test_disagreement_fails(self):
        mf = MetaFilter(min_models_agree=2, max_disagreement=0.15)
        preds = {
            "model_a": {"home": 0.70, "draw": 0.20, "away": 0.10},
            "model_b": {"home": 0.30, "draw": 0.40, "away": 0.30},
            "model_c": {"home": 0.65, "draw": 0.20, "away": 0.15},
        }
        # model_b doesn't agree (draw is its best), only 2 agree on home
        # but std of home probs [0.70, 0.30, 0.65] ≈ 0.18 > 0.15
        result = mf.check_agreement(preds, "home")
        assert result.passed is False

    def test_no_agreement_fails(self):
        mf = MetaFilter(min_models_agree=2)
        preds = {
            "model_a": {"home": 0.60, "draw": 0.25, "away": 0.15},
            "model_b": {"home": 0.20, "draw": 0.50, "away": 0.30},
        }
        result = mf.check_agreement(preds, "home")
        assert result.passed is False
        assert "agree" in result.reason.lower()


# ---------------------------------------------------------------------------
# FilterResult tests
# ---------------------------------------------------------------------------

class TestFilterResult:
    """Test FilterResult dataclass."""

    def test_pass(self):
        r = FilterResult(True)
        assert r.passed is True
        assert r.reason is None

    def test_fail_with_reason(self):
        r = FilterResult(False, reason="Too risky")
        assert r.passed is False
        assert r.reason == "Too risky"
