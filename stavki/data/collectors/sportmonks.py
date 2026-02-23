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
from pathlib import Path
import threading
from stavki.config import PROJECT_ROOT


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

    
    def __init__(
        self,
        api_key: str,
        rate_limit: int = 180,  # Requests per minute
        timeout: Any = (3.05, 27), # (connect, read) tuple
        cache_ttl: int = 300,  # Cache TTL in seconds
        retries: int = 5       # Max retries
    ):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.max_retries = retries
        
        # Load leagues from config
        self.league_ids = {}
        try:
            # Try loading from league_config.json first (source of truth)
            # Try loading from stavki/config/leagues.json
            config_path = PROJECT_ROOT / "stavki" / "config" / "leagues.json"
            if config_path.exists():
                with open(config_path) as f:
                    full_config = json.load(f)
                    
                    # 1. New Format: per_league -> league -> sportmonks_id
                    per_league = full_config.get("per_league", {})
                    for league_key, league_data in per_league.items():
                        if "sportmonks_id" in league_data:
                            self.league_ids[league_key] = league_data["sportmonks_id"]
                            simple_name = league_key.replace("soccer_", "").upper()
                            self.league_ids[simple_name] = league_data["sportmonks_id"]
                            
                    # 2. Legacy/Simple Format: EPL: 8 directly at top level or in explicit map
                    # Check top level keys that look like league names
                    for k, v in full_config.items():
                        if isinstance(v, int):
                            self.league_ids[k] = v
                        elif k == "league_ids" and isinstance(v, dict):
                            self.league_ids.update(v)
            
            if not self.league_ids: 
                logger.warning("No league IDs found in config, falling back to legacy defaults")
                self.league_ids = {
                    "EPL": 8, "LA_LIGA": 564, "BUNDESLIGA": 82,
                    "SERIE_A": 384, "LIGUE_1": 301, "CHAMPIONSHIP": 9,
                }
        except Exception as e:
            logger.error(f"Failed to load leagues config: {e}")
            self.league_ids = {
                "EPL": 8, "LA_LIGA": 564, "BUNDESLIGA": 82,
                "SERIE_A": 384, "LIGUE_1": 301, "CHAMPIONSHIP": 9,
            }
            
        # Reverse mapping for display
        self.league_names = {v: k for k, v in self.league_ids.items()}
        
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
        
        # Thread safety lock for rate limiting queue
        self._rate_limit_lock = threading.RLock()
        
        logger.info("SportMonks client initialized")

    
    def _rate_limit_wait(self):
        """Enforce rate limiting with per-request throttle across multiple threads."""
        with self._rate_limit_lock:
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
                # After sleeping, update 'now' to reflect the current time
                now = time.time()
            
            self._request_times.append(now)
            self._last_request_time = now
    
    def _request(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        includes: Optional[List[str]] = None,
        max_retries: Optional[int] = None,
    ) -> Dict:
        """Make API request with rate limiting, retry on 429, and error handling."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        # Use instance default if not provided
        if max_retries is None:
            max_retries = self.max_retries
            
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
        # SportMonks v3 stat type_id mapping
        TYPE_ID_MAP = {
            42: "shots-total",
            86: "shots-on-target",
            45: "ball-possession",
            34: "corners",
            56: "fouls",
            84: "yellowcards",
            83: "redcards",
            580: "big-chances",
            49: "shots-inside",
            50: "shots-outside",
        }
        
        result = MatchStats(fixture_id=fixture_id)
        
        # Temp storage for proxy calculation
        proxy_stats = {
            "home": {"big-chances": 0, "shots-inside": 0, "shots-outside": 0, "shots-total": 0},
            "away": {"big-chances": 0, "shots-inside": 0, "shots-outside": 0, "shots-total": 0}
        }
        
        for stat in stats:
            location = stat.get("location")  # "home" or "away"
            if location not in ["home", "away"]: 
                continue
                
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
            
            # Helper for safe int/float conversion
            def get_val(v, is_float=False):
                if v is None: return None
                return float(v) if is_float else int(v)

            if code == "shots-total":
                val = get_val(value)
                if location == "home": result.home_shots = val
                else: result.away_shots = val
                proxy_stats[location]["shots-total"] = val or 0
                
            elif code == "shots-on-target":
                val = get_val(value)
                if location == "home": result.home_shots_on_target = val
                else: result.away_shots_on_target = val
                
            elif code == "ball-possession":
                val = get_val(value, True)
                if location == "home": result.home_possession = val
                else: result.away_possession = val
                
            elif code == "corners":
                val = get_val(value)
                if location == "home": result.home_corners = val
                else: result.away_corners = val
                
            elif code == "fouls":
                val = get_val(value)
                if location == "home": result.home_fouls = val
                else: result.away_fouls = val
                
            elif code == "yellowcards":
                val = get_val(value)
                if location == "home": result.home_yellow_cards = val
                else: result.away_yellow_cards = val
                
            elif code == "redcards":
                val = get_val(value)
                if location == "home": result.home_red_cards = val
                else: result.away_red_cards = val
                
            elif code == "expected-goals":
                val = get_val(value, True)
                if location == "home": result.home_xg = val
                else: result.away_xg = val
            
            # Derived stats for Proxy xG
            elif code == "big-chances":
                proxy_stats[location]["big-chances"] = get_val(value) or 0
            elif code == "shots-inside":
                proxy_stats[location]["shots-inside"] = get_val(value) or 0
            elif code == "shots-outside":
                proxy_stats[location]["shots-outside"] = get_val(value) or 0
        
        # Calculate Proxy xG if missing
        for side in ["home", "away"]:
            current_xg = getattr(result, f"{side}_xg")
            if current_xg is None:
                # Proxy Formula: same as RealXGBuilder
                # xG ≈ (Big Chances * 0.45) + (Shots Inside * 0.08) + (Shots Outside * 0.03)
                
                stats_side = proxy_stats[side]
                bc = stats_side["big-chances"]
                inside = max(0, stats_side["shots-inside"] - bc)
                outside = stats_side["shots-outside"]
                
                proxy_xg = (bc * 0.45) + (inside * 0.08) + (outside * 0.03)
                
                if proxy_xg == 0 and stats_side["shots-total"] > 0:
                    # Fallback coarse: 0.10 per shot
                     proxy_xg = stats_side["shots-total"] * 0.10
                
                if proxy_xg > 0:
                     setattr(result, f"{side}_xg", round(proxy_xg, 2))
        
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
                "coaches"
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
                    "coaches"
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
        
    def get_multiple_fixtures_full(self, fixture_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Get all data for multiple fixtures in ONE API call via multi endpoint.
        
        Returns:
            Dict mapping fixture_id -> Dict with keys: stats, lineups, referee, events, weather, participants, coaches
        """
        if not fixture_ids:
            return {}
            
        id_str = ",".join(map(str, fixture_ids))
        response = self._request(
            f"fixtures/multi/{id_str}",
            includes=[
                "statistics",
                "lineups.player",
                "events",
                "referees.referee",
                "coaches",
                "participants",
                "weather"
            ]
        )
        
        data_list = response.get("data", [])
        if not isinstance(data_list, list):
            data_list = []
            
        results = {}
        for data in data_list:
            fix_id = data.get("id")
            if not fix_id:
                continue
                
            match_data = {
                "fixture_id": fix_id,
                "stats": None,
                "lineups": [],
                "referee": None,
                "events": data.get("events", []),
                "participants": data.get("participants", []),
                "coaches": data.get("coaches", []),
                "weather": data.get("weather", {})
            }
            
            # --- Parse statistics ---
            stats_data = data.get("statistics", [])
            if stats_data:
                match_data["stats"] = self._parse_statistics(fix_id, stats_data)
            
            # --- Parse referee ---
            refs = data.get("referees", [])
            if refs:
                main_ref = next((r for r in refs if r.get("type_id") == 6), refs[0])
                ref_detail = main_ref.get("referee", {})
                if isinstance(ref_detail, dict):
                    ref_name = ref_detail.get("common_name") or ref_detail.get("name")
                else:
                    ref_name = None
                if ref_name:
                    match_data["referee"] = ref_name
                    if match_data["stats"]:
                        match_data["stats"].referee_name = ref_name
            
            # --- Lineups ---
            match_data["lineups"] = data.get("lineups", [])
            
            results[fix_id] = match_data
            
        return results
        
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
        
        # Handle case where weather is not a dict (e.g. "NS" string or None)
        if not weather or not isinstance(weather, dict):
            return None
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
            normalized_name = market_name.lower()
            target_market_type = "1x2" # default
            
            if "Result & Both Teams To Score" in market_name or "Result/Both Teams To Score" in market_name:
                match_found = True
                target_market_type = "result_btts"
            elif "Correct Score" in market_name:
                match_found = True
                target_market_type = "correct_score"
            elif "Double Chance" in market_name:
                match_found = True
                target_market_type = "double_chance"
            
            # Existing checks...
            elif market == "1X2":
                # Primary 1X2 Market
                valid_aliases = ["1x2", "fulltime result", "match winner", "3way result", "match result"]
                # Exclude ONLY if we are specifically looking for PURE 1X2
                # If we are in "ALL" mode (which we are), we want everything properly labeled.
                exclude_terms = ["corner", "card", "half", "period", "handicap", "booking", "goal", "qualify"]
                
                if any(alias in normalized_name for alias in valid_aliases):
                    if not any(ex in normalized_name for ex in exclude_terms):
                        # Ensure we don't accidentally grab "Result & BTTS" as "1x2" here
                        # The specific check above handles "Result & BTTS", so if we are here, likely safe
                        if "btts" not in normalized_name and "both teams" not in normalized_name:
                            match_found = True
                            target_market_type = "1x2"
                        
            elif market == "ALL":
                # We want everything valid
                # 1. Main 1X2
                if any(alias in normalized_name for alias in ["1x2", "fulltime result", "match winner"]):
                    if not any(ex in normalized_name for ex in ["corner", "card", "half", "period"]):
                        if "btts" in normalized_name or "both teams" in normalized_name:
                             match_found = True
                             target_market_type = "result_btts"
                        else:
                             match_found = True
                             target_market_type = "1x2"
                
                # 2. Corners 1X2 
                elif "corner" in normalized_name and "1x2" in normalized_name:
                    match_found = True
                    target_market_type = "corners_1x2"
                    
                # 3. BTTS (Both Teams To Score)
                elif "both teams to score" in normalized_name or "btts" in normalized_name:
                     if "result" not in normalized_name and "winner" not in normalized_name:
                         match_found = True
                         target_market_type = "btts"
                         
                # 4. Correct Score
                elif "correct score" in normalized_name:
                    match_found = True
                    target_market_type = "correct_score"
                    
                # 5. Double Chance
                elif "double chance" in normalized_name:
                    match_found = True
                    target_market_type = "double_chance"

            if not match_found:
                continue
            
            # Key by bookmaker AND market type
            bm_id = odd.get("bookmaker_id")
            if not bm_id:
                continue
            
            # Create unique key for grouping (Bookmaker + Market)
            group_key = f"{bm_id}_{target_market_type}"

            if group_key not in grouped_odds:
                grouped_odds[group_key] = {
                    "bookmaker": odd.get("bookmaker", {}).get("name", "Unknown"),
                    "market": market_name,
                    "market_type": target_market_type, 
                    "timestamp": odd.get("updated_at") or odd.get("latest_bookmaker_update"),
                    "odds": {}
                }
            
            # Extract Value
            label = odd.get("label", "").lower()
            value = odd.get("value")
            if not value:
                continue

            # Standardize labels based on market type
            std_label = label # Default: keep original label (e.g. "2-1", "Draw / Yes")
            
            if target_market_type == "1x2" or target_market_type == "corners_1x2":
                if "1" == label or "home" in label: std_label = "home"
                elif "x" == label or "draw" in label: std_label = "draw"
                elif "2" == label or "away" in label: std_label = "away"
            
            elif target_market_type == "btts":
                if "yes" in label: std_label = "yes"
                elif "no" in label: std_label = "no"
                
            try:
                grouped_odds[group_key]["odds"][std_label] = float(value)
            except (ValueError, TypeError):
                continue
        
        return list(grouped_odds.values())
                

    
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
        if league == League.EPL: sm_league_id = self.client.league_ids.get("EPL")
        elif league == League.LA_LIGA: sm_league_id = self.client.league_ids.get("LA_LIGA")
        elif league == League.BUNDESLIGA: sm_league_id = self.client.league_ids.get("BUNDESLIGA")
        elif league == League.SERIE_A: sm_league_id = self.client.league_ids.get("SERIE_A")
        elif league == League.LIGUE_1: sm_league_id = self.client.league_ids.get("LIGUE_1")
        elif league == League.CHAMPIONSHIP: sm_league_id = self.client.league_ids.get("CHAMPIONSHIP")
        
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
                # Fetch ALL key markets (1X2, BTTS, Corners)
                sm_odds = self.client.get_fixture_odds(int(m.source_id), market="ALL")
                
                snapshots = []
                for o in sm_odds:
                    market_type = o.get("market_type", "1x2")
                    odds_map = o.get("odds", {})
                    
                    # Map odds based on market type
                    home = None
                    draw = None
                    away = None
                    
                    # 1. Standard 1X2 / Corners 1X2 / Double Chance (Home/Draw/Away structure)
                    if market_type in ["1x2", "corners_1x2", "double_chance"]:
                        home = odds_map.get("home") or odds_map.get("1") or odds_map.get("1x")
                        draw = odds_map.get("draw") or odds_map.get("x") or odds_map.get("12") # 12 is usually Home/Away but DC is weird
                        away = odds_map.get("away") or odds_map.get("2") or odds_map.get("x2")
                        
                        # Double Chance special mapping if needed, but usually APIs normalize to 1X, X2, 12
                        # SportMonks labels: "1X", "X2", "12"
                        if market_type == "double_chance":
                             home = odds_map.get("1x") or odds_map.get("Home/Draw")
                             draw = odds_map.get("12") or odds_map.get("Home/Away") # Middle option often
                             away = odds_map.get("x2") or odds_map.get("Draw/Away")

                    # 2. BTTS (Yes/No)
                    elif market_type == "btts":
                        home = odds_map.get("yes")
                        away = odds_map.get("no")
                        draw = None # No draw in BTTS
                    
                    # 3. Complex Markets (Result & BTTS, Correct Score)
                    # We store them but might abuse the fields or need a new schema property.
                    # For now, we will store the raw odds_map in a metadata field if permitted, 
                    # OR we just store the most relevant ones.
                    # Actually, for "Result & BTTS", we have outcomes like "Home & Yes", "Draw & Yes", etc.
                    # schema.OddsSnapshot expects home/draw/away floats. 
                    # We need to extend the schema or pack data.
                    # Current schema: home_odds, draw_odds, away_odds.
                    # Let's map "Home & Yes" -> home_odds, "Draw & Yes" -> draw_odds, "Away & Yes" -> away_odds
                    # determining the specific sub-market (Yes or No) might be needed.
                    # Simplification: Store "Yes" combo in the main fields (Home/Draw/Away & Yes)
                    
                    elif market_type == "result_btts":
                        # Attempt to find "Home & Yes", "Draw & Yes", "Away & Yes"
                        # Labels might be: "1 & Yes", "X & Yes", "2 & Yes"
                        home = odds_map.get("Home & Yes")
                        draw = odds_map.get("Draw & Yes")
                        away = odds_map.get("Away & Yes")
                    
                    elif market_type == "correct_score":
                        # Too many outcomes for 3 columns. match_id/bookmaker unique constraint usually exists.
                        # We can't fit 20 scores into home/draw/away.
                        # We skip Correct Score for now in the Snapshot schema unless we refactor it.
                        continue

                    # Validation: Must have at least meaningful odds
                    # For BTTS, we need Home(Yes) and Away(No). Draw is None.
                    if (home and away) or (market_type == "result_btts" and home and away):
                        snap = OddsSnapshot(
                            match_id=m.id,
                            bookmaker=o.get("bookmaker", "Unknown"),
                            timestamp=datetime.fromisoformat(o.get("timestamp").replace("Z", "+00:00")) if o.get("timestamp") else datetime.now(),
                            home_odds=home,
                            draw_odds=draw,
                            away_odds=away,
                            market=market_type  # Store the specific market type
                        )
                        snapshots.append(snap)
                    else:
                        pass # Skip incomplete odds
                
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

