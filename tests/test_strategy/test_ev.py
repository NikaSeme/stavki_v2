"""
Tests for strategy.ev — Expected Value calculator
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from stavki.strategy.ev import EVCalculator, EVResult, compute_ev


class TestCalculateEV:
    """Test the core EV formula: EV = p * odds - 1."""

    def test_positive_ev(self):
        # 50% chance at 2.5 odds → EV = 0.5 * 2.5 - 1 = 0.25
        assert EVCalculator.calculate_ev(0.50, 2.5) == pytest.approx(0.25)

    def test_negative_ev(self):
        # 30% chance at 2.0 odds → EV = 0.3 * 2.0 - 1 = -0.4
        assert EVCalculator.calculate_ev(0.30, 2.0) == pytest.approx(-0.4)

    def test_zero_ev(self):
        # Fair odds: 50% at 2.0 → EV = 0
        assert EVCalculator.calculate_ev(0.50, 2.0) == pytest.approx(0.0)

    def test_edge_prob_zero(self):
        assert EVCalculator.calculate_ev(0.0, 2.0) == -1.0

    def test_edge_prob_one(self):
        assert EVCalculator.calculate_ev(1.0, 2.0) == -1.0

    def test_edge_odds_one(self):
        assert EVCalculator.calculate_ev(0.5, 1.0) == -1.0

    def test_edge_odds_below_one(self):
        assert EVCalculator.calculate_ev(0.5, 0.5) == -1.0


class TestImpliedProb:
    """Test odds → implied probability conversion."""

    def test_evens(self):
        assert EVCalculator.implied_prob(2.0) == pytest.approx(0.5)

    def test_heavy_favorite(self):
        assert EVCalculator.implied_prob(1.25) == pytest.approx(0.8)

    def test_longshot(self):
        assert EVCalculator.implied_prob(10.0) == pytest.approx(0.1)

    def test_invalid_odds(self):
        assert EVCalculator.implied_prob(1.0) == 0.0
        assert EVCalculator.implied_prob(0.5) == 0.0


class TestRemoveVig:
    """Test bookmaker margin removal."""

    def test_overround_removal(self):
        # Typical 1X2 market with ~5% overround
        odds = {"home": 2.0, "draw": 3.5, "away": 4.0}
        fair = EVCalculator.remove_vig(odds)

        # Fair probs should sum to exactly 1.0
        total = sum(fair.values())
        assert total == pytest.approx(1.0, abs=0.001)

        # Each fair prob should be lower than implied (since vig is removed)
        for selection in odds:
            implied = 1.0 / odds[selection]
            assert fair[selection] <= implied

    def test_fair_market(self):
        # Already fair (no vig)
        odds = {"home": 2.0, "draw": 5.0, "away": 10.0 / 3.0}
        fair = EVCalculator.remove_vig(odds)
        assert sum(fair.values()) == pytest.approx(1.0, abs=0.001)

    def test_empty_odds(self):
        assert EVCalculator.remove_vig({}) == {}


class TestEvaluateBet:
    """Test the full evaluate_bet pipeline."""

    def test_positive_ev_passes(self):
        calc = EVCalculator(min_ev=0.03)
        result = calc.evaluate_bet(
            match_id="test_001",
            market="1X2",
            selection="home",
            model_prob=0.60,
            odds=2.5,
        )
        assert result is not None
        assert result.ev > 0.03
        assert result.match_id == "test_001"

    def test_low_ev_filtered(self):
        calc = EVCalculator(min_ev=0.03)
        result = calc.evaluate_bet(
            match_id="test_002",
            market="1X2",
            selection="home",
            model_prob=0.40,
            odds=2.5,  # EV = 0.0, below threshold
        )
        assert result is None

    def test_prob_below_min_filtered(self):
        calc = EVCalculator(min_prob=0.10)
        result = calc.evaluate_bet(
            match_id="test_003",
            market="1X2",
            selection="home",
            model_prob=0.05,  # below min_prob
            odds=25.0,
        )
        assert result is None

    def test_odds_outside_range_filtered(self):
        calc = EVCalculator(max_odds=10.0)
        result = calc.evaluate_bet(
            match_id="test_004",
            market="1X2",
            selection="home",
            model_prob=0.50,
            odds=15.0,  # above max_odds
        )
        assert result is None

    def test_with_vig_removal(self):
        calc = EVCalculator(min_ev=0.03, use_no_vig=True)
        all_odds = {"home": 2.0, "draw": 3.5, "away": 4.0}
        result = calc.evaluate_bet(
            match_id="test_005",
            market="1X2",
            selection="home",
            model_prob=0.65,
            odds=2.0,
            all_odds=all_odds,
        )
        if result:
            # With vig removal, implied prob should be lower than 1/odds
            assert result.implied_prob < 1.0 / 2.0


class TestComputeEV:
    """Test standalone compute_ev function."""

    def test_delegates_to_calculator(self):
        assert compute_ev(0.5, 2.5) == pytest.approx(
            EVCalculator.calculate_ev(0.5, 2.5)
        )


class TestEVResult:
    """Test EVResult dataclass."""

    def test_is_value_positive(self):
        r = EVResult(
            match_id="m1", market="1X2", selection="home",
            model_prob=0.6, odds=2.5, ev=0.5, edge_pct=0.1, implied_prob=0.4,
        )
        assert r.is_value is True

    def test_is_value_negative(self):
        r = EVResult(
            match_id="m1", market="1X2", selection="home",
            model_prob=0.3, odds=2.0, ev=-0.4, edge_pct=-0.2, implied_prob=0.5,
        )
        assert r.is_value is False

    def test_to_dict(self):
        r = EVResult(
            match_id="m1", market="1X2", selection="home",
            model_prob=0.6, odds=2.5, ev=0.5, edge_pct=0.1, implied_prob=0.4,
        )
        d = r.to_dict()
        assert d["match_id"] == "m1"
        assert d["ev"] == 0.5
