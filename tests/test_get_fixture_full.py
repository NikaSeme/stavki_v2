"""
Regression test for get_fixture_full().

Verifies that get_fixture_full() returns a dict with the required keys
and correct types. This test would FAIL on the broken version where
the parsing code was orphaned after get_multiple_fixtures_full().
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from stavki.data.collectors.sportmonks import SportMonksClient


# Realistic API response for fixtures/{id} with includes
MOCK_FIXTURE_RESPONSE = {
    "data": {
        "id": 12345,
        "league_id": 8,
        "starting_at": "2025-01-15T15:00:00Z",
        "participants": [
            {"id": 100, "name": "Arsenal", "meta": {"location": "home"}},
            {"id": 200, "name": "Chelsea", "meta": {"location": "away"}},
        ],
        "statistics": [
            {
                "location": "home",
                "type_id": 42,
                "data": {"value": 15},
            },
            {
                "location": "away",
                "type_id": 42,
                "data": {"value": 10},
            },
        ],
        "referees": [
            {
                "type_id": 6,
                "referee": {"common_name": "Michael Oliver", "name": "M. Oliver"},
            }
        ],
        "lineups": [
            {"team_id": 100, "player_id": 1, "player_name": "Raya",
             "type_id": 11, "formation_position": 1, "jersey_number": 22,
             "formation_field": "1:1"},
        ],
        "events": [
            {
                "type_id": 14,
                "minute": 23,
                "player_name": "Saka",
                "participant_id": 100,
                "result": "1-0",
            }
        ],
    }
}


@pytest.fixture
def client():
    """Create a SportMonksClient with mocked _request."""
    with patch.object(SportMonksClient, "__init__", lambda self, *a, **kw: None):
        c = SportMonksClient.__new__(SportMonksClient)
        # Set required attributes that __init__ would create
        c._session = MagicMock()
        c._last_request_time = 0
        c._min_request_interval = 0.0
        c._lock = MagicMock()
        c.PLAYER_DETAIL_IDS = SportMonksClient.PLAYER_DETAIL_IDS
        return c


class TestGetFixtureFullRegression:
    """Regression: get_fixture_full must never return None."""

    def test_returns_dict_not_none(self, client):
        """The broken code returned None because parsing was dead code."""
        with patch.object(client, "_request", return_value=MOCK_FIXTURE_RESPONSE):
            result = client.get_fixture_full(12345)

        assert result is not None, (
            "get_fixture_full() returned None — parsing code is still orphaned"
        )
        assert isinstance(result, dict), (
            f"Expected dict, got {type(result).__name__}"
        )

    def test_required_keys_present(self, client):
        """All required keys must be in the returned dict."""
        with patch.object(client, "_request", return_value=MOCK_FIXTURE_RESPONSE):
            result = client.get_fixture_full(12345)

        required_keys = {"fixture_id", "stats", "lineups", "referee", "events"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_fixture_id_matches(self, client):
        """fixture_id must match the requested id."""
        with patch.object(client, "_request", return_value=MOCK_FIXTURE_RESPONSE):
            result = client.get_fixture_full(12345)

        assert result["fixture_id"] == 12345

    def test_stats_parsed(self, client):
        """Statistics must be parsed into a MatchStats object (not None/raw)."""
        with patch.object(client, "_request", return_value=MOCK_FIXTURE_RESPONSE):
            result = client.get_fixture_full(12345)

        stats = result["stats"]
        assert stats is not None, "stats should not be None when API returns statistics"
        assert hasattr(stats, "home_shots"), "stats should be a MatchStats object"
        assert stats.home_shots == 15
        assert stats.away_shots == 10

    def test_referee_parsed(self, client):
        """Referee name must be extracted from referees include."""
        with patch.object(client, "_request", return_value=MOCK_FIXTURE_RESPONSE):
            result = client.get_fixture_full(12345)

        assert result["referee"] == "Michael Oliver"

    def test_events_parsed(self, client):
        """Events must be a non-empty list of parsed dicts."""
        with patch.object(client, "_request", return_value=MOCK_FIXTURE_RESPONSE):
            result = client.get_fixture_full(12345)

        events = result["events"]
        assert isinstance(events, list)
        assert len(events) == 1
        assert events[0]["type"] == "goal"
        assert events[0]["minute"] == 23

    def test_lineups_is_dict_or_list(self, client):
        """lineups must be a dict (keyed by home/away) or a list."""
        with patch.object(client, "_request", return_value=MOCK_FIXTURE_RESPONSE):
            result = client.get_fixture_full(12345)

        lineups = result["lineups"]
        assert isinstance(lineups, (dict, list)), (
            f"lineups should be dict or list, got {type(lineups).__name__}"
        )

    def test_empty_api_response(self, client):
        """Empty API response must still return a valid dict with all keys."""
        empty_response = {"data": {}}
        with patch.object(client, "_request", return_value=empty_response):
            result = client.get_fixture_full(99999)

        assert isinstance(result, dict)
        assert result["fixture_id"] == 99999
        assert result["stats"] is None
        assert result["events"] == []

    def test_fallback_on_403(self, client):
        """On 403 error from detailed lineups, must retry and still return dict."""
        error_response = {"data": [], "error": "forbidden"}

        def side_effect(endpoint, includes=None, **kw):
            # First call with lineups.details.type -> forbidden
            if includes and "lineups.details.type" in includes:
                return error_response
            # Fallback call without details -> OK
            return {"data": {"id": 12345, "statistics": [], "lineups": [],
                             "events": [], "referees": []}}

        with patch.object(client, "_request", side_effect=side_effect):
            result = client.get_fixture_full(12345)

        assert isinstance(result, dict)
        assert result["fixture_id"] == 12345


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
