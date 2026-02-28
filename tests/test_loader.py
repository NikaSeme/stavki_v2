"""
Tests for UnifiedDataLoader, with focused coverage on _fetch_recent_from_api
and its caller path in get_historical_data.

Contract gate requirement:
  - callee-level regression tests for _fetch_recent_from_api
  - caller-path test proving get_historical_data invokes _fetch_recent_from_api
"""

import pytest
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, PropertyMock
import json
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.data.loader import UnifiedDataLoader, DataSource
from stavki.data.collectors.sportmonks import MatchFixture, MatchStats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir(tmp_path):
    """Create temp directory for cache/data."""
    d = tmp_path / "data"
    d.mkdir()
    (d / "raw" / "epl").mkdir(parents=True)
    return d


@pytest.fixture
def loader(temp_dir):
    """Initialize loader with temp cache and mocked client."""
    with patch("stavki.data.loader.PROJECT_ROOT", temp_dir.parent):
        ldr = UnifiedDataLoader(
            api_key="test_key",
            cache_dir=temp_dir / ".cache"
        )
    # Override leagues_map to a known value for deterministic tests
    ldr.leagues_map = {8: "epl", 82: "bundesliga"}
    return ldr


def _make_fixture(fixture_id=1001, league_id=8,
                  home="Arsenal", away="Chelsea"):
    """Helper: build a MatchFixture for mocking."""
    return MatchFixture(
        fixture_id=fixture_id,
        league_id=league_id,
        home_team=home,
        home_team_id=100,
        away_team=away,
        away_team_id=200,
        kickoff=datetime(2025, 1, 15, 15, 0),
    )


def _make_stats(fixture_id=1001):
    """Helper: build a MatchStats stub with xG and shots."""
    return MatchStats(
        fixture_id=fixture_id,
        home_xg=1.5,
        away_xg=0.8,
        home_shots=12,
        away_shots=7,
        home_possession=58.0,
        away_possession=42.0,
    )


# =========================================================================
# EXISTING TESTS  (normalization, cache, historical)
# =========================================================================

class TestUnifiedDataLoader:

    def test_normalization(self, loader):
        """Test team name normalization returns a string."""
        # normalize_team_name delegates to centralized_normalize.
        # We test the delegation contract, not the synonym table itself.
        result = loader.normalize_team_name("Arsenal")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_caching(self, loader):
        """Test cache read/write."""
        key = "test_key"
        data = [{"id": 1, "name": "Test"}]

        # Write
        loader._set_cache(key, data)

        # Read
        cached = loader._get_cached(key)
        assert cached == data

        # Test missing
        assert loader._get_cached("missing") is None

    def test_historical_data_mock(self, loader, temp_dir):
        """Test that get_historical_data loads CSV files from data/raw and returns DataFrame."""
        # Create a minimal CSV for the loader to find
        csv_dir = temp_dir.parent / "data" / "raw" / "epl"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "epl_2024_25.csv"
        csv_path.write_text(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
            "15/01/2025,Arsenal,Chelsea,2,1,H,1.8,3.5,4.2\n"
        )

        with patch("stavki.data.loader.PROJECT_ROOT", temp_dir.parent):
            result = loader.get_historical_data(
                start_date="2025-01-01",
                end_date="2025-01-31",
                force_reload=True,
            )

        assert isinstance(result, pd.DataFrame), (
            f"Expected pd.DataFrame, got {type(result).__name__}"
        )
        assert not result.empty, "DataFrame should not be empty when CSV exists"
        assert "HomeTeam" in result.columns, "Must contain HomeTeam column"
        assert "AwayTeam" in result.columns, "Must contain AwayTeam column"
        assert result.iloc[0]["HomeTeam"] == "Arsenal"


# =========================================================================
# _fetch_recent_from_api — CALLEE REGRESSION TESTS
# =========================================================================

class TestFetchRecentFromApi:
    """Callee-level regression tests for _fetch_recent_from_api()."""

    def test_returns_dataframe_always(self, loader):
        """_fetch_recent_from_api must always return pd.DataFrame."""
        loader.client = MagicMock()
        loader.client.get_fixtures_by_date.return_value = []

        result = loader._fetch_recent_from_api(
            start_date="2025-01-15", end_date="2025-01-15"
        )

        assert isinstance(result, pd.DataFrame), (
            f"Expected pd.DataFrame, got {type(result).__name__}"
        )

    def test_returns_empty_df_when_no_client(self, loader):
        """Without an API client, must return empty DataFrame immediately."""
        loader.client = None

        result = loader._fetch_recent_from_api("2025-01-15", "2025-01-15")

        assert isinstance(result, pd.DataFrame)
        assert result.empty, "Expected empty DataFrame when client is None"

    def test_returns_cached_data_if_available(self, loader):
        """If cache hit, must return cached DataFrame without calling API."""
        cached_rows = [
            {"Date": "2025-01-15", "HomeTeam": "Arsenal", "AwayTeam": "Chelsea", "League": "epl"}
        ]
        loader._set_cache("recent_2025-01-15_2025-01-15", cached_rows)
        loader.client = MagicMock()

        result = loader._fetch_recent_from_api("2025-01-15", "2025-01-15")

        assert len(result) == 1
        assert result.iloc[0]["HomeTeam"] == "Arsenal"
        loader.client.get_fixtures_by_date.assert_not_called()

    def test_basic_fixture_fetch(self, loader):
        """Happy path: one fixture, no odds/stats errors."""
        fixture = _make_fixture()
        loader.client = MagicMock()
        loader.client.get_fixtures_by_date.return_value = [fixture]
        loader.client.get_fixture_odds.return_value = [
            {"odds": {"home": 2.0, "draw": 3.5, "away": 4.0}}
        ]
        loader.client.get_fixture_full.return_value = {
            "fixture_id": 1001,
            "stats": _make_stats(),
            "lineups": {},
            "referee": "Michael Oliver",
            "events": [],
        }

        result = loader._fetch_recent_from_api("2025-01-15", "2025-01-15")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["HomeTeam"] == "Arsenal"
        assert row["AwayTeam"] == "Chelsea"
        assert row["League"] == "epl"
        assert row["B365H"] == 2.0
        assert row["xG_home"] == 1.5
        assert row["referee"] == "Michael Oliver"

    def test_get_fixture_full_non_dict_guard(self, loader):
        """If get_fixture_full returns non-dict, enrichment must be skipped (not crash)."""
        fixture = _make_fixture()
        loader.client = MagicMock()
        loader.client.get_fixtures_by_date.return_value = [fixture]
        loader.client.get_fixture_odds.return_value = []
        # Simulate old broken code returning None
        loader.client.get_fixture_full.return_value = None

        result = loader._fetch_recent_from_api("2025-01-15", "2025-01-15")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        # No xG columns because enrichment was skipped
        assert "xG_home" not in result.columns

    def test_api_exception_does_not_crash(self, loader):
        """API exceptions must be caught, returning partial/empty DataFrame."""
        loader.client = MagicMock()
        loader.client.get_fixtures_by_date.side_effect = RuntimeError("API down")

        result = loader._fetch_recent_from_api("2025-01-15", "2025-01-15")

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_multi_day_range(self, loader):
        """Must iterate day-by-day across the date range."""
        fixture_day1 = _make_fixture(fixture_id=1, home="Arsenal", away="Chelsea")
        fixture_day2 = _make_fixture(fixture_id=2, home="Liverpool", away="Everton")

        def by_date(date_str, league_ids=None):
            if date_str == "2025-01-15":
                return [fixture_day1]
            elif date_str == "2025-01-16":
                return [fixture_day2]
            return []

        loader.client = MagicMock()
        loader.client.get_fixtures_by_date.side_effect = by_date
        loader.client.get_fixture_odds.return_value = []
        loader.client.get_fixture_full.return_value = {"fixture_id": 0, "stats": None, "lineups": {}, "referee": None, "events": []}

        result = loader._fetch_recent_from_api("2025-01-15", "2025-01-16")

        assert len(result) == 2
        assert set(result["HomeTeam"]) == {"Arsenal", "Liverpool"}

    def test_odds_exception_graceful(self, loader):
        """Odds failure must not prevent fixture from appearing in output."""
        fixture = _make_fixture()
        loader.client = MagicMock()
        loader.client.get_fixtures_by_date.return_value = [fixture]
        loader.client.get_fixture_odds.side_effect = RuntimeError("odds API down")
        loader.client.get_fixture_full.return_value = {"fixture_id": 1001, "stats": None, "lineups": {}, "referee": None, "events": []}

        result = loader._fetch_recent_from_api("2025-01-15", "2025-01-15")

        assert len(result) == 1
        assert "B365H" not in result.columns  # odds not set


# =========================================================================
# CALLER-PATH TESTS — get_historical_data → _fetch_recent_from_api
# =========================================================================

class TestCallerPathFetchRecentFromApi:
    """Caller-path tests: verify get_historical_data invokes _fetch_recent_from_api."""

    def test_get_historical_data_calls_fetch_recent(self, loader, temp_dir):
        """get_historical_data(include_recent_from_api=True) must call _fetch_recent_from_api."""
        # Return a DF with 'Date' so sort_values('Date') won't crash
        stub_df = pd.DataFrame({"Date": ["2025-01-15"], "HomeTeam": ["A"], "AwayTeam": ["B"]})
        with patch.object(loader, "_fetch_recent_from_api", return_value=stub_df) as mock_fetch:
            with patch("stavki.data.loader.PROJECT_ROOT", temp_dir.parent):
                result = loader.get_historical_data(
                    start_date="2025-01-01",
                    end_date="2025-01-31",
                    include_recent_from_api=True,
                )
            mock_fetch.assert_called_once()
            # The returned DF should contain data from the fetch
            assert not result.empty

    def test_get_historical_data_no_api_when_flag_false(self, loader, temp_dir):
        """Without include_recent_from_api, _fetch_recent_from_api must NOT be called."""
        # Create a minimal CSV so get_historical_data has data and doesn't crash on empty sort
        csv_dir = temp_dir.parent / "data" / "raw" / "epl"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "epl_2024_25.csv"
        csv_path.write_text(
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
            "15/01/2025,Arsenal,Chelsea,2,1,H,1.8,3.5,4.2\n"
        )
        with patch.object(loader, "_fetch_recent_from_api") as mock_fetch:
            with patch("stavki.data.loader.PROJECT_ROOT", temp_dir.parent):
                # Force reload so it reads our CSV
                loader.get_historical_data(
                    start_date="2025-01-01",
                    end_date="2025-01-31",
                    include_recent_from_api=False,
                    force_reload=True,
                )
            mock_fetch.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
