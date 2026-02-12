"""
Tests for strategy.kelly — Kelly criterion staking
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from stavki.strategy.kelly import KellyStaker, StakeResult, kelly_simple
from stavki.strategy.ev import EVResult


# ---------------------------------------------------------------------------
# Kelly formula tests
# ---------------------------------------------------------------------------

class TestKellyFormula:
    """Test the core Kelly formula: f* = (b*p - q) / b."""

    def test_positive_edge(self):
        # 60% chance at 2.0 (even money): f = (1*0.6 - 0.4) / 1 = 0.2
        assert KellyStaker.kelly_formula(0.60, 2.0) == pytest.approx(0.2)

    def test_no_edge(self):
        # 50% at 2.0 → f = 0.0
        assert KellyStaker.kelly_formula(0.50, 2.0) == pytest.approx(0.0)

    def test_negative_edge(self):
        # 30% at 2.0 → f = (1*0.3 - 0.7) / 1 = -0.4 → clamped to 0
        assert KellyStaker.kelly_formula(0.30, 2.0) == 0.0

    def test_high_odds(self):
        # 15% at 10.0: f = (9*0.15 - 0.85) / 9 ≈ 0.0556
        result = KellyStaker.kelly_formula(0.15, 10.0)
        assert result == pytest.approx(0.0556, abs=0.001)

    def test_edge_cases(self):
        assert KellyStaker.kelly_formula(0.0, 2.0) == 0.0
        assert KellyStaker.kelly_formula(1.0, 2.0) == 0.0
        assert KellyStaker.kelly_formula(0.5, 1.0) == 0.0
        assert KellyStaker.kelly_formula(0.5, 0.5) == 0.0


# ---------------------------------------------------------------------------
# KellyStaker.calculate_stake tests
# ---------------------------------------------------------------------------

def _make_ev_result(prob=0.60, odds=2.5, ev=0.50, edge=0.10) -> EVResult:
    """Helper to create an EVResult."""
    return EVResult(
        match_id="test_match",
        market="1X2",
        selection="home",
        model_prob=prob,
        odds=odds,
        ev=ev,
        edge_pct=edge,
        implied_prob=1.0 / odds,
    )


class TestCalculateStake:
    """Test stake calculation with limits."""

    def test_positive_kelly_produces_stake(self):
        staker = KellyStaker(bankroll=1000.0)
        ev = _make_ev_result(prob=0.60, odds=2.5)
        result = staker.calculate_stake(ev, apply_limits=False)
        assert result.stake_pct > 0
        assert result.stake_amount > 0
        assert result.kelly_full > 0

    def test_negative_kelly_zero_stake(self):
        staker = KellyStaker(bankroll=1000.0)
        ev = _make_ev_result(prob=0.30, odds=2.0, ev=-0.4)
        result = staker.calculate_stake(ev)
        assert result.stake_pct == 0.0
        assert result.stake_amount == 0.0
        assert result.reason == "Negative Kelly"

    def test_fractional_kelly_reduces_stake(self):
        staker = KellyStaker(bankroll=1000.0, config={"kelly_fraction": 0.25})
        ev = _make_ev_result(prob=0.60, odds=2.5)
        result = staker.calculate_stake(ev, apply_limits=False)

        full_kelly = KellyStaker.kelly_formula(0.60, 2.5)
        expected_pct = full_kelly * 0.25
        assert result.stake_pct == pytest.approx(expected_pct, abs=0.001)

    def test_limits_cap_stake(self):
        staker = KellyStaker(bankroll=1000.0, config={"max_stake_pct": 0.01})
        ev = _make_ev_result(prob=0.80, odds=3.0)  # Very high Kelly
        result = staker.calculate_stake(ev, apply_limits=True)
        assert result.stake_pct <= 0.01


# ---------------------------------------------------------------------------
# Settle bet tests
# ---------------------------------------------------------------------------

class TestSettleBet:
    """Test bet settlement (win/loss)."""

    def test_win_increases_bankroll(self):
        staker = KellyStaker(bankroll=1000.0)
        ev = _make_ev_result(prob=0.60, odds=2.5)
        stake = staker.calculate_stake(ev, league="EPL", apply_limits=False)
        staker.place_bet(stake, league="EPL")

        initial_bankroll = staker.bankroll
        staker.settle_bet("test_match", "win")
        assert staker.bankroll > initial_bankroll

    def test_loss_decreases_bankroll(self):
        staker = KellyStaker(bankroll=1000.0)
        ev = _make_ev_result(prob=0.60, odds=2.5)
        stake = staker.calculate_stake(ev, league="EPL", apply_limits=False)
        staker.place_bet(stake, league="EPL")

        initial_bankroll = staker.bankroll
        staker.settle_bet("test_match", "loss")
        assert staker.bankroll < initial_bankroll

    def test_void_no_bankroll_change(self):
        staker = KellyStaker(bankroll=1000.0)
        ev = _make_ev_result(prob=0.60, odds=2.5)
        stake = staker.calculate_stake(ev, league="EPL", apply_limits=False)
        staker.place_bet(stake, league="EPL")

        initial_bankroll = staker.bankroll
        staker.settle_bet("test_match", "void")
        assert staker.bankroll == pytest.approx(initial_bankroll)


# ---------------------------------------------------------------------------
# kelly_simple tests
# ---------------------------------------------------------------------------

class TestKellySimple:
    """Test standalone kelly_simple function."""

    def test_quarter_kelly(self):
        full = KellyStaker.kelly_formula(0.60, 2.0)
        assert kelly_simple(0.60, 2.0, fraction=0.25) == pytest.approx(
            full * 0.25
        )

    def test_zero_edge(self):
        assert kelly_simple(0.50, 2.0) == 0.0
