"""
SportMonks Football API Client.

Provides access to:
- Match fixtures and results
- xG (Expected Goals) statistics
- Team lineups and formations
- Player injuries and suspensions
- Live scores and events
- Pre-match and in-play odds
- Weather forecasts for matches
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


@dataclass
class MatchFixture:
    """Parsed match fixture data."""
    fixture_id: int
    league_id: int
    home_team: str
    home_team_id: int
    away_team: str
    away_team_id: int
    kickoff: datetime
    venue: Optional[str] = None
    round: Optional[str] = None
    status: str = "NS"  # Not Started


@dataclass
class MatchStats:
    """Match statistics including xG."""
    fixture_id: int
    home_xg: Optional[float] = None
    away_xg: Optional[float] = None
    home_shots: Optional[int] = None
    away_shots: Optional[int] = None
    home_shots_on_target: Optional[int] = None
    away_shots_on_target: Optional[int] = None
    home_possession: Optional[float] = None
    away_possession: Optional[float] = None
    home_corners: Optional[int] = None
    away_corners: Optional[int] = None
    home_fouls: Optional[int] = None
    away_fouls: Optional[int] = None
    home_yellow_cards: Optional[int] = None
    away_yellow_cards: Optional[int] = None
    home_red_cards: Optional[int] = None
    away_red_cards: Optional[int] = None
    referee_name: Optional[str] = None


@dataclass 
class TeamLineup:
    """Team lineup data."""
    team_id: int
    team_name: str
    formation: Optional[str]
    starting_xi: List[Dict]
    substitutes: List[Dict]
    coach: Optional[str] = None


@dataclass
class InjuryInfo:
    """Player injury/suspension info."""
    player_id: int
    player_name: str
    team_id: int
    team_name: str
    type: str  # "injury" or "suspension"
    reason: Optional[str] = None
    expected_return: Optional[datetime] = None


class SportMonksClient:
    """
    SportMonks Football API v3 Client.
    
    Features:
    - Automatic rate limiting
    - Response caching
    - Error handling with retries
    - Data parsing into typed objects
    
    Usage:
        client = SportMonksClient(api_key="your_key")
        fixtures = client.get_fixtures_by_date("2024-01-15")
        stats = client.get_fixture_stats(fixture_id=123456)
    """
    
    BASE_URL = "https://api.sportmonks.com/v3/football"
    
    # European Advanced league IDs
    LEAGUE_IDS = {
        "EPL": 8,
        "LA_LIGA": 564,
        "BUNDESLIGA": 82,
        "SERIE_A": 384,
        "LIGUE_1": 301,
        "CHAMPIONSHIP": 9,
        "EREDIVISIE": 72,
        "PRIMEIRA": 462,
        "SCOTTISH": 501,
        "BELGIAN": 208,
    }
    
    # Reverse mapping for display
    LEAGUE_NAMES = {v: k for k, v in LEAGUE_IDS.items()}
    
    def __init__(
        self,
        api_key: str,
        rate_limit: int = 180,  # Requests per minute
        timeout: int = 30,
        cache_ttl: int = 300  # Cache TTL in seconds
    ):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        
        # Rate limiting
        self._request_times: List[float] = []
        self._last_request_time: float = 0.0
        self._min_request_interval: float = 1.2  # seconds between requests (50/min safe)
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": api_key,
            "Accept": "application/json"
        })
        
        logger.info("SportMonks client initialized")
    
    def _rate_limit_wait(self):
        """Enforce rate limiting with per-request throttle."""
        # Minimum delay between requests to prevent burst-triggering
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        
        now = time.time()
        minute_ago = now - 60
        
        # Remove old requests
        self._request_times = [t for t in self._request_times if t > minute_ago]
        
        # Wait if at limit
        if len(self._request_times) >= self.rate_limit:
            wait_time = self._request_times[0] - minute_ago + 0.1
            logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        self._request_times.append(now)
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        includes: Optional[List[str]] = None,
        max_retries: int = 5,
    ) -> Dict:
        """Make API request with rate limiting, retry on 429, and error handling."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Build params
        request_params = params or {}
        if includes:
            request_params["include"] = ";".join(includes)
        
        for attempt in range(max_retries + 1):
            self._rate_limit_wait()
            
            try:
                response = self.session.get(
                    url,
                    params=request_params,
                    timeout=self.timeout
                )
                self._last_request_time = time.time()
                
                # Handle 429 with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries:
                        wait = min(2 ** (attempt + 1), 60)
                        logger.warning(f"Rate limited (429), retry {attempt+1}/{max_retries} in {wait}s")
                        # Adaptive: increase throttle temporarily
                        self._min_request_interval = min(self._min_request_interval * 1.5, 5.0)
                        time.sleep(wait)
                        continue
                    else:
                        logger.error(f"Rate limited after {max_retries} retries: {endpoint}")
                        # Long pause to let rate window fully reset
                        logger.info("Pausing 120s to let rate limit window reset...")
                        time.sleep(120)
                        # Reset throttle to moderate level
                        self._min_request_interval = 2.0
                        return {"data": [], "error": "rate_limited"}
                
                # Handle 403 Forbidden (include not available)
                if response.status_code == 403:
                    logger.debug(f"Forbidden (403): {endpoint}")
                    return {"data": [], "error": "forbidden"}
                
                response.raise_for_status()
                
                data = response.json()
                
                # Check for API errors
                if "error" in data:
                    logger.error(f"SportMonks API error: {data['error']}")
                    return {"data": [], "error": data["error"]}
                
                return data
                
            except requests.exceptions.Timeout:
                logger.error(f"Request timeout: {endpoint}")
                return {"data": [], "error": "timeout"}
            except requests.exceptions.RequestException as e:
                if "429" in str(e) and attempt < max_retries:
                    wait = min(2 ** (attempt + 1), 60)
                    logger.warning(f"Rate limited (429), retry {attempt+1}/{max_retries} in {wait}s")
                    time.sleep(wait)
                    continue
                logger.error(f"Request failed: {e}")
                return {"data": [], "error": str(e)}
        
        return {"data": [], "error": "max_retries_exceeded"}
    
    # =========================================================================
    # FIXTURES
    # =========================================================================
    
    def get_fixtures_by_date(
        self,
        date: str,
        league_ids: Optional[List[int]] = None
    ) -> List[MatchFixture]:
        """
        Get all fixtures for a specific date (handles pagination).
        
        Args:
            date: Date in YYYY-MM-DD format
            league_ids: Optional list of league IDs to filter (client-side)
            
        Returns:
            List of MatchFixture objects
        """
        endpoint = f"fixtures/date/{date}"
        
        all_fixtures = []
        page = 1
        
        while True:
            params = {"page": str(page)}
            
            response = self._request(
                endpoint,
                params=params,
                includes=["participants", "venue", "round"]
            )
            
            data_items = response.get("data", [])
            if not data_items:
                break
            
            for item in data_items:
                try:
                    participants = item.get("participants", [])
                    home = next((p for p in participants if p.get("meta", {}).get("location") == "home"), None)
                    away = next((p for p in participants if p.get("meta", {}).get("location") == "away"), None)
                    
                    if home and away:
                        fixture = MatchFixture(
                            fixture_id=item["id"],
                            league_id=item.get("league_id"),
                            home_team=home.get("name", "Unknown"),
                            home_team_id=home.get("id"),
                            away_team=away.get("name", "Unknown"),
                            away_team_id=away.get("id"),
                            kickoff=datetime.fromisoformat(item["starting_at"].replace("Z", "+00:00")),
                            venue=item.get("venue", {}).get("name") if item.get("venue") else None,
                            round=item.get("round", {}).get("name") if item.get("round") else None,
                            status=item.get("state", {}).get("short", "NS")
                        )
                        all_fixtures.append(fixture)
                except Exception as e:
                    logger.warning(f"Failed to parse fixture {item.get('id')}: {e}")
            
            # Check if there's a next page
            pagination = response.get("pagination", {})
            has_more = pagination.get("has_more", False)
            if not has_more:
                break
            page += 1
        
        # Client-side league filtering (more reliable than API filter)
        if league_ids:
            league_set = set(league_ids)
            all_fixtures = [f for f in all_fixtures if f.league_id in league_set]
        
        logger.info(f"Got {len(all_fixtures)} fixtures for {date}")
        return all_fixtures
    
    def get_upcoming_fixtures(
        self,
        days: int = 7,
        league_ids: Optional[List[int]] = None
    ) -> List[MatchFixture]:
        """Get fixtures for the next N days."""
        all_fixtures = []
        
        for i in range(days):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d")
            fixtures = self.get_fixtures_by_date(date, league_ids)
            all_fixtures.extend(fixtures)
        
        return all_fixtures
    
    # =========================================================================
    # STATISTICS & xG
    # =========================================================================
    
    def get_fixture_stats(self, fixture_id: int) -> Optional[MatchStats]:
        """
        Get detailed statistics for a fixture including xG.
        
        Args:
            fixture_id: SportMonks fixture ID
            
        Returns:
            MatchStats object or None
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["statistics"]
        )
        
        data = response.get("data", {})
        stats = data.get("statistics", [])
        
        if not stats:
            return None
        
        return self._parse_statistics(fixture_id, stats)
    
    def _parse_statistics(self, fixture_id: int, stats: list) -> MatchStats:
        """
        Parse statistics list into MatchStats object.
        
        Handles both formats:
        - type_id integers (from 'statistics' include)
        - nested type.code strings (from 'statistics.type' include)
        """
        # SportMonks v3 stat type_id mapping
        TYPE_ID_MAP = {
            42: "shots-total",
            86: "shots-on-target",
            45: "ball-possession",
            34: "corners",
            56: "fouls",
            84: "yellowcards",
            83: "redcards",
        }
        
        result = MatchStats(fixture_id=fixture_id)
        
        for stat in stats:
            location = stat.get("location")  # "home" or "away"
            value = stat.get("data", {}).get("value")
            
            # Determine stat code from type_id or nested type object
            type_id = stat.get("type_id")
            code = TYPE_ID_MAP.get(type_id, "")
            if not code:
                type_obj = stat.get("type", {})
                if isinstance(type_obj, dict):
                    code = type_obj.get("code", "")
                    if "expected_goals" in type_obj.get("name", "").lower():
                        code = "expected-goals"
            
            if not code:
                continue
            
            if code == "shots-total":
                if location == "home":
                    result.home_shots = int(value) if value is not None else None
                else:
                    result.away_shots = int(value) if value is not None else None
            elif code == "shots-on-target":
                if location == "home":
                    result.home_shots_on_target = int(value) if value is not None else None
                else:
                    result.away_shots_on_target = int(value) if value is not None else None
            elif code == "ball-possession":
                if location == "home":
                    result.home_possession = float(value) if value is not None else None
                else:
                    result.away_possession = float(value) if value is not None else None
            elif code == "corners":
                if location == "home":
                    result.home_corners = int(value) if value is not None else None
                else:
                    result.away_corners = int(value) if value is not None else None
            elif code == "fouls":
                if location == "home":
                    result.home_fouls = int(value) if value is not None else None
                else:
                    result.away_fouls = int(value) if value is not None else None
            elif code == "yellowcards":
                if location == "home":
                    result.home_yellow_cards = int(value) if value is not None else None
                else:
                    result.away_yellow_cards = int(value) if value is not None else None
            elif code == "redcards":
                if location == "home":
                    result.home_red_cards = int(value) if value is not None else None
                else:
                    result.away_red_cards = int(value) if value is not None else None
            elif code == "expected-goals":
                if location == "home":
                    result.home_xg = float(value) if value is not None else None
                else:
                    result.away_xg = float(value) if value is not None else None
        
        return result
    
    # Key detail type_ids from lineups.details.type
    PLAYER_DETAIL_IDS = {
        118: "rating",
        119: "minutes_played",
        42: "shots",
        86: "shots_on_target",
        117: "key_passes",
        580: "big_chances_created",
        581: "big_chances_missed",
        120: "touches",
        1584: "accurate_passes_pct",
        52: "goals",
        79: "assists",
    }
    
    def _parse_player_entry(self, p: dict) -> dict:
        """Parse a single player lineup entry with per-player details."""
        entry = {
            "id": p.get("player_id"),
            "name": p.get("player_name") or (
                p.get("player", {}).get("display_name")
                if isinstance(p.get("player"), dict) else None
            ),
            "position": p.get("formation_position"),
            "jersey": p.get("jersey_number"),
        }
        
        # Parse per-player details (from lineups.details.type include)
        details = p.get("details", [])
        if details:
            for d in details:
                type_id = d.get("type_id")
                key = self.PLAYER_DETAIL_IDS.get(type_id)
                if key:
                    value = d.get("data", {}).get("value")
                    if value is not None:
                        entry[key] = value
        
        return entry
    
    def get_fixture_full(self, fixture_id: int) -> Dict[str, Any]:
        """
        Get all data for a fixture in ONE API call.
        
        Tries lineups.details.type for per-player stats (rating, minutes, etc.)
        Falls back to plain lineups if the include returns 403.
        
        Returns:
            Dict with keys: stats, lineups, referee, events
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=[
                "statistics",
                "lineups.details.type",
                "events",
                "referees.referee",
            ]
        )
        
        data = response.get("data", {})
        
        # Fallback: if data is a list or response has error (403 from details include),
        # retry without per-player details
        if isinstance(data, list) or response.get("error"):
            response = self._request(
                f"fixtures/{fixture_id}",
                includes=[
                    "statistics",
                    "lineups",
                    "events",
                    "referees.referee",
                ]
            )
            data = response.get("data", {})
            if isinstance(data, list):
                data = {}
        
        result = {
            "fixture_id": fixture_id,
            "stats": None,
            "lineups": {},
            "referee": None,
            "events": [],
        }
        
        # --- Parse statistics ---
        stats_data = data.get("statistics", [])
        if stats_data:
            result["stats"] = self._parse_statistics(fixture_id, stats_data)
        
        # --- Parse referee (type_id=6 is main referee in v3) ---
        refs = data.get("referees", [])
        if refs:
            main_ref = next((r for r in refs if r.get("type_id") == 6), refs[0])
            ref_detail = main_ref.get("referee", {})
            if isinstance(ref_detail, dict):
                ref_name = ref_detail.get("common_name") or ref_detail.get("name")
            else:
                ref_name = None
            if ref_name:
                result["referee"] = ref_name
                if result["stats"]:
                    result["stats"].referee_name = ref_name
        
        # --- Parse lineups (v3: flat array of player entries) ---
        # type_id=11 = starting XI, type_id=12 = substitutes
        lineups_data = data.get("lineups", [])
        if lineups_data:
            from collections import Counter
            team_ids = sorted(set(l.get("team_id") for l in lineups_data))
            
            # Determine home/away from participants
            participants = data.get("participants", [])
            home_tid = away_tid = None
            for p in participants:
                loc = p.get("meta", {}).get("location")
                if loc == "home":
                    home_tid = p.get("id")
                elif loc == "away":
                    away_tid = p.get("id")
            
            for team_id in team_ids:
                team_players = [l for l in lineups_data if l.get("team_id") == team_id]
                
                has_starters = any(p.get("type_id") == 11 for p in team_players)
                
                if has_starters:
                    # Modern format: type_id distinguishes starters vs subs
                    starting = [p for p in team_players if p.get("type_id") == 11]
                    subs = [p for p in team_players if p.get("type_id") == 12]
                else:
                    # Legacy format: all type_id=12, first 11 are starters
                    starting = team_players[:11]
                    subs = team_players[11:]
                
                # Determine location
                if team_id == home_tid:
                    location = "home"
                elif team_id == away_tid:
                    location = "away"
                else:
                    location = "home" if team_id == team_ids[0] else "away"
                
                # Derive formation from formation_field values
                formation = None
                fields = [p.get("formation_field") for p in starting if p.get("formation_field")]
                if fields:
                    rows = Counter(f.split(":")[0] for f in fields if ":" in f)
                    parts = [str(rows[r]) for r in sorted(rows.keys()) if r != "1"]
                    if parts:
                        formation = "-".join(parts)
                
                team_lineup = TeamLineup(
                    team_id=team_id,
                    team_name="Unknown",
                    formation=formation,
                    starting_xi=[self._parse_player_entry(p) for p in starting],
                    substitutes=[self._parse_player_entry(p) for p in subs],
                    coach=None,
                )
                result["lineups"][location] = team_lineup
        
        # --- Parse events (v3: type_id integers) ---
        EVENT_TYPE_MAP = {
            14: "goal", 15: "own-goal", 16: "penalty-goal",
            17: "penalty-miss", 18: "substitution",
            19: "yellowcard", 20: "redcard", 21: "yellowredcard",
        }
        events_data = data.get("events", [])
        for event in events_data:
            parsed = {
                "type": EVENT_TYPE_MAP.get(event.get("type_id"), str(event.get("type_id", ""))),
                "minute": event.get("minute"),
                "player": event.get("player_name"),
                "team_id": event.get("participant_id"),
                "result": event.get("result"),
            }
            result["events"].append(parsed)
        
        return result
    
    def get_team_xg_stats(
        self,
        team_id: int,
        season_id: Optional[int] = None,
        last_n_matches: int = 10
    ) -> Dict[str, float]:
        """
        Get aggregated xG statistics for a team.
        
        Returns:
            Dict with xg_for, xg_against, xg_diff averages
        """
        # Get recent fixtures for team
        response = self._request(
            f"teams/{team_id}/fixtures",
            params={"per_page": last_n_matches},
            includes=["statistics"]
        )
        
        fixtures = response.get("data", [])
        
        xg_for_list = []
        xg_against_list = []
        
        for fixture in fixtures:
            stats = fixture.get("statistics", [])
            for stat in stats:
                if "expected_goals" in str(stat.get("type", {}).get("name", "")).lower():
                    participant_id = stat.get("participant_id")
                    value = stat.get("data", {}).get("value")
                    
                    if value:
                        if participant_id == team_id:
                            xg_for_list.append(float(value))
                        else:
                            xg_against_list.append(float(value))
        
        return {
            "xg_for_avg": sum(xg_for_list) / len(xg_for_list) if xg_for_list else 0.0,
            "xg_against_avg": sum(xg_against_list) / len(xg_against_list) if xg_against_list else 0.0,
            "xg_diff_avg": (sum(xg_for_list) - sum(xg_against_list)) / max(len(xg_for_list), 1),
            "matches_analyzed": len(xg_for_list)
        }
    
    # =========================================================================
    # LINEUPS & INJURIES
    # =========================================================================
    
    def get_fixture_lineups(self, fixture_id: int) -> Dict[str, TeamLineup]:
        """
        Get confirmed lineups for a fixture.
        
        Returns:
            Dict with "home" and "away" TeamLineup objects
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["lineups.player", "lineups.details"]
        )
        
        data = response.get("data", {})
        lineups_data = data.get("lineups", [])
        
        result = {}
        
        for lineup in lineups_data:
            team_id = lineup.get("team_id")
            location = lineup.get("meta", {}).get("location", "home")
            
            players = lineup.get("lineup", [])
            starting = [p for p in players if p.get("meta", {}).get("position", 0) <= 11]
            subs = [p for p in players if p.get("meta", {}).get("position", 0) > 11]
            
            team_lineup = TeamLineup(
                team_id=team_id,
                team_name=lineup.get("team", {}).get("name", "Unknown"),
                formation=lineup.get("formation", {}).get("formation"),
                starting_xi=[{
                    "id": p.get("player_id"),
                    "name": p.get("player", {}).get("display_name"),
                    "position": p.get("meta", {}).get("position"),
                    "jersey": p.get("jersey_number")
                } for p in starting],
                substitutes=[{
                    "id": p.get("player_id"),
                    "name": p.get("player", {}).get("display_name"),
                    "jersey": p.get("jersey_number")
                } for p in subs],
                coach=lineup.get("coach", {}).get("name") if lineup.get("coach") else None
            )
            
            result[location] = team_lineup
        
        return result
    
    def get_team_injuries(self, team_id: int) -> List[InjuryInfo]:
        """
        Get current injuries and suspensions for a team.
        
        Returns:
            List of InjuryInfo objects
        """
        response = self._request(
            f"teams/{team_id}",
            includes=["sidelined.player", "sidelined.type"]
        )
        
        data = response.get("data", {})
        sidelined = data.get("sidelined", [])
        
        injuries = []
        for item in sidelined:
            # Check if current (no end date or end date in future)
            end_date = item.get("end_date")
            if end_date:
                try:
                    end_dt = datetime.fromisoformat(end_date)
                    if end_dt < datetime.now():
                        continue  # Already returned
                except (ValueError, TypeError) as e:
                    logger.debug(f"Could not parse end_date '{end_date}': {e}")
            
            injury = InjuryInfo(
                player_id=item.get("player_id"),
                player_name=item.get("player", {}).get("display_name", "Unknown"),
                team_id=team_id,
                team_name=data.get("name", "Unknown"),
                type="suspension" if "suspension" in str(item.get("type", {}).get("name", "")).lower() else "injury",
                reason=item.get("type", {}).get("name"),
                expected_return=datetime.fromisoformat(end_date) if end_date else None
            )
            injuries.append(injury)
        
        return injuries
    
    # =========================================================================
    # WEATHER
    # =========================================================================
    
    def get_fixture_weather(self, fixture_id: int) -> Optional[Dict]:
        """
        Get weather forecast for a fixture.
        
        Returns:
            Dict with temperature, precipitation, wind, humidity
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["weatherreport"]
        )
        
        data = response.get("data", {})
        weather = data.get("weatherreport", {})
        
        if not weather:
            return None
        
        return {
            "temperature": weather.get("temperature", {}).get("temp"),
            "feels_like": weather.get("temperature", {}).get("feels_like"),
            "humidity": weather.get("humidity"),
            "wind_speed": weather.get("wind", {}).get("speed"),
            "wind_direction": weather.get("wind", {}).get("degree"),
            "clouds": weather.get("clouds", {}).get("all"),
            "condition": weather.get("description"),
            "icon": weather.get("icon")
        }
    
    # =========================================================================
    # ODDS
    # =========================================================================
    
    def get_fixture_odds(
        self,
        fixture_id: int,
        market: str = "1X2"
    ) -> List[Dict]:
        """
        Get pre-match odds for a fixture.
        
        Args:
            fixture_id: Fixture ID
            market: Market type ("1X2", "Over/Under", etc.)
            
        Returns:
            List of odds from different bookmakers
        """
        response = self._request(
            f"fixtures/{fixture_id}",
            includes=["odds.bookmaker", "odds.market"]
        )
        
        data = response.get("data", {})
        all_odds = data.get("odds", [])
        
        # Group by bookmaker since V3 returns flat list of outcomes
        grouped_odds = {}
        
        for odd in all_odds:
            market_name = odd.get("market", {}).get("name", "")
            logger.debug(f"Checking odds market: '{market_name}' against target '{market}'")
            
            # Smart market matching
            match_found = False
            if market == "1X2":
                if any(alias in market_name for alias in ["1X2", "Fulltime Result", "Match Winner", "3Way Result"]):
                    match_found = True
            else:
                if market.lower() in market_name.lower():
                    match_found = True
                    
            if not match_found:
                continue
            
            # Key by bookmaker
            bm_id = odd.get("bookmaker_id")
            if not bm_id:
                continue
                
            if bm_id not in grouped_odds:
                grouped_odds[bm_id] = {
                    "bookmaker": odd.get("bookmaker", {}).get("name", "Unknown"),
                    "market": market_name,
                    "timestamp": odd.get("updated_at") or odd.get("latest_bookmaker_update"),
                    "odds": {}
                }
            
            # Extract Value
            label = odd.get("label", "").lower()
            value = odd.get("value")
            if not value:
                continue
                
            try:
                val_float = float(value)
                if label in ["1", "home"]:
                    grouped_odds[bm_id]["odds"]["home"] = val_float
                elif label in ["x", "draw"]:
                    grouped_odds[bm_id]["odds"]["draw"] = val_float
                elif label in ["2", "away"]:
                    grouped_odds[bm_id]["odds"]["away"] = val_float
            except ValueError:
                continue
        
        # Convert groups to list
        result = []
        for data in grouped_odds.values():
            # Only include if we have at least one outcome (or strictly all 3?)
            # Schema expects all 3 usually, but let's be permissive and filter later if needed.
            # Actually, fetch_odds checks for all 3.
            if data["odds"]:
                result.append(data)
        
        return result
    
    # =========================================================================
    # UTILITY
    # =========================================================================
    
    def test_connection(self) -> bool:
        """Test API connection and key validity."""
        try:
            response = self._request("leagues", params={"per_page": 1})
            return "data" in response and not response.get("error")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status."""
        minute_ago = time.time() - 60
        recent_requests = len([t for t in self._request_times if t > minute_ago])
        
        return {
            "requests_last_minute": recent_requests,
            "limit": self.rate_limit,
            "remaining": self.rate_limit - recent_requests
        }


# =========================================================================
# COLLECTOR WRAPPER
# =========================================================================

class SportMonksCollector:
    """
    High-level collector for SportMonks data.
    
    Adapts SportMonksClient to the common collector interface.
    """
    
    def __init__(self, client: Optional[SportMonksClient] = None):
        if client:
            self.client = client
        else:
            from ...config import get_config
            config = get_config()
            self.client = SportMonksClient(api_key=config.sportmonks_api_key)
            
        from ..processors.validate import OddsValidator
        self.validator = OddsValidator()
        
    def fetch_matches(
        self,
        league: "League",
        include_odds: bool = True,
        max_hours_ahead: int = 48
    ) -> List["Match"]:
        """
        Fetch upcoming matches for a league.
        """
        from ..schemas import Match, Team, League
        from ..processors.normalize import SourceNormalizer
        
        # Map our League enum to SportMonks ID
        sm_league_id = None
        if league == League.EPL: sm_league_id = SportMonksClient.LEAGUE_IDS["EPL"]
        elif league == League.LA_LIGA: sm_league_id = SportMonksClient.LEAGUE_IDS["LA_LIGA"]
        elif league == League.BUNDESLIGA: sm_league_id = SportMonksClient.LEAGUE_IDS["BUNDESLIGA"]
        elif league == League.SERIE_A: sm_league_id = SportMonksClient.LEAGUE_IDS["SERIE_A"]
        elif league == League.LIGUE_1: sm_league_id = SportMonksClient.LEAGUE_IDS["LIGUE_1"]
        elif league == League.CHAMPIONSHIP: sm_league_id = SportMonksClient.LEAGUE_IDS["CHAMPIONSHIP"]
        
        if not sm_league_id:
            logger.warning(f"SportMonks ID not found for league: {league}")
            return []
            
        # Calc days needed
        days = (max_hours_ahead // 24) + 1
        fixtures = self.client.get_upcoming_fixtures(days=days, league_ids=[sm_league_id])
        
        matches = []
        cutoff = datetime.now() + timedelta(hours=max_hours_ahead)
        
        for f in fixtures:
            # Skip if too far ahead
            if f.kickoff > cutoff:
                continue
                
            try:
                match = Match(
                    id=str(f.fixture_id),
                    home_team=Team(
                        name=f.home_team,
                        normalized_name=SourceNormalizer.from_sportmonks(f.home_team),
                    ),
                    away_team=Team(
                        name=f.away_team,
                        normalized_name=SourceNormalizer.from_sportmonks(f.away_team),
                    ),
                    league=league,
                    commence_time=f.kickoff,
                    source="sportmonks",
                    source_id=str(f.fixture_id),
                )
                matches.append(match)
            except Exception as e:
                logger.warning(f"Failed to convert fixture {f.fixture_id}: {e}")
                
        return matches

    def fetch_odds(
        self,
        league: "League",
        matches: Optional[List["Match"]] = None
    ) -> Dict[str, List["OddsSnapshot"]]:
        """
        Fetch odds for given matches or league.
        Note: SportMonks odds fetching is per-fixture, so passing 'matches' is preferred
        to avoid re-fetching the schedule.
        """
        from ..schemas import OddsSnapshot
        
        if not matches:
            matches = self.fetch_matches(league, max_hours_ahead=48)
            
        results = {}
        
        for m in matches:
            try:
                sm_odds = self.client.get_fixture_odds(int(m.source_id), market="1X2")
                
                snapshots = []
                for o in sm_odds:
                    # Construct snapshot
                    odds_map = o.get("odds", {})
                    if "home" in odds_map and "draw" in odds_map and "away" in odds_map:
                        snap = OddsSnapshot(
                            match_id=m.id,
                            bookmaker=o.get("bookmaker", "Unknown"),
                            timestamp=datetime.fromisoformat(o.get("timestamp").replace("Z", "+00:00")) if o.get("timestamp") else datetime.now(),
                            home_odds=odds_map["home"],
                            draw_odds=odds_map["draw"],
                            away_odds=odds_map["away"]
                        )
                        snapshots.append(snap)
                    else:
                        missing = [k for k in ["home", "draw", "away"] if k not in odds_map]
                        logger.debug(f"Dropped snapshot for match {m.id} (Bookmaker: {o.get('bookmaker', 'Unknown')}): Missing {missing}")
                
                results[m.id] = snapshots
                
            except Exception as e:
                logger.warning(f"Failed to fetch odds for match {m.id}: {e}")
        
        return results

    def fetch_match_details(
        self, 
        match: "Match"
    ) -> Tuple[Optional["MatchStats"], Optional["MatchLineups"]]:
        """
        Fetch detailed stats (xG) and lineups for a completed or live match.
        """
        from ..schemas import MatchStats, MatchLineups, TeamLineup, Player
        
        if not match.source_id or match.source != "sportmonks":
            # If we don't have a specific source ID, we can't fetch details easily
            # (Unless we search by similarity, but let's assume we have it)
            return None, None
            
        try:
            source_id = int(match.source_id)
        except ValueError:
            return None, None
        
        # Use single API call instead of separate stats + lineups calls
        full_data = self.client.get_fixture_full(source_id)
        
        # 1. Stats
        sm_stats = full_data.get("stats")
        match_stats = None
        if sm_stats:
            match_stats = MatchStats(
                match_id=match.id,
                xg_home=sm_stats.home_xg,
                xg_away=sm_stats.away_xg,
                possession_home=sm_stats.home_possession,
                possession_away=sm_stats.away_possession,
                shots_home=sm_stats.home_shots,
                shots_away=sm_stats.away_shots,
                shots_on_target_home=sm_stats.home_shots_on_target,
                shots_on_target_away=sm_stats.away_shots_on_target,
                corners_home=sm_stats.home_corners,
                corners_away=sm_stats.away_corners,
            )
            
        # 2. Lineups (from same API call)
        sm_lineups = full_data.get("lineups", {})
        match_lineups = None
        if sm_lineups:
            home_l = sm_lineups.get("home")
            away_l = sm_lineups.get("away")
            
            if home_l and away_l:
                def convert_lineup(sm_l):
                    return TeamLineup(
                        team_name=sm_l.team_name,
                        formation=sm_l.formation,
                        starting_xi=[
                            Player(
                                id=str(p.get("id", "")), 
                                name=p.get("name", ""), 
                                position=str(p.get("position") or ""), 
                                rating=p.get("rating")
                            ) for p in sm_l.starting_xi
                        ],
                        substitutes=[
                            Player(
                                id=str(p.get("id", "")), 
                                name=p.get("name", ""), 
                                position=str(p.get("position") or ""), 
                                rating=p.get("rating")
                            ) for p in sm_l.substitutes
                        ]
                    )
                    
                match_lineups = MatchLineups(
                    match_id=match.id,
                    home=convert_lineup(home_l),
                    away=convert_lineup(away_l)
                )
                
        return match_stats, match_lineups

