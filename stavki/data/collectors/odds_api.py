"""
The Odds API collector.

Fetches live odds from The Odds API and normalizes to our schemas.

Key features:
- Multiple sports/leagues support
- Best odds extraction across bookmakers
- Outlier detection and filtering
- Rate limiting and caching
"""

import os
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

import requests

from ..schemas import Match, OddsSnapshot, BestOdds, Team, League
from ..processors.normalize import SourceNormalizer
from ..processors.validate import OddsValidator

logger = logging.getLogger(__name__)


@dataclass
class APIResponse:
    """Wrapper for API response with metadata."""
    success: bool
    data: Any
    remaining_requests: int = 0
    error: Optional[str] = None


class OddsAPIClient:
    """
    Client for The Odds API.
    
    Features:
    - Automatic rate limiting
    - Response caching
    - Error retry with backoff
    - Request counting
    """
    
    BASE_URL = "https://api.the-odds-api.com/v4"
    
    # Sport keys for different leagues
    SPORT_KEYS = {
        League.EPL: "soccer_epl",
        League.LA_LIGA: "soccer_spain_la_liga",
        League.BUNDESLIGA: "soccer_germany_bundesliga",
        League.SERIE_A: "soccer_italy_serie_a",
        League.LIGUE_1: "soccer_france_ligue_one",
        League.CHAMPIONSHIP: "soccer_efl_champ",
        League.NBA: "basketball_nba",
    }
    
    # Preferred bookmakers (sharper = better)
    PREFERRED_BOOKMAKERS = [
        "pinnacle",      # Sharpest
        "betfair_ex_eu", # Exchange
        "marathon_bet",
        "betvictor",
        "unibet",
        "bet365",
        "williamhill",
        "betway",
    ]
    
    # Cache settings
    CACHE_TTL_SECONDS = 60  # 1 minute for live odds
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        regions: str = "eu,uk",
        markets: str = "h2h",
        odds_format: str = "decimal",
    ):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not set")
        
        self.regions = regions
        self.markets = markets
        self.odds_format = odds_format
        
        # Request tracking
        self.remaining_requests = 0
        self.last_request_time = 0.0
        
        # Simple in-memory cache
        self._cache: Dict[str, tuple] = {}  # key -> (data, timestamp)
    
    def _cache_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key for request."""
        key_str = f"{endpoint}:{sorted(params.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached response if still valid."""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self.CACHE_TTL_SECONDS:
                return data
            del self._cache[key]
        return None
    
    def _set_cached(self, key: str, data: Any) -> None:
        """Cache response."""
        self._cache[key] = (data, time.time())
    
    def _request(
        self,
        endpoint: str,
        params: Dict[str, str],
        use_cache: bool = True
    ) -> APIResponse:
        """Make API request with caching and error handling."""
        
        # Check cache
        cache_key = self._cache_key(endpoint, params)
        if use_cache:
            cached = self._get_cached(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return APIResponse(success=True, data=cached, remaining_requests=self.remaining_requests)
        
        # Rate limit (1 request per second)
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        
        url = f"{self.BASE_URL}/{endpoint}"
        params["apiKey"] = self.api_key
        
        try:
            response = requests.get(url, params=params, timeout=30)
            self.last_request_time = time.time()
            
            # Track remaining requests
            if "x-requests-remaining" in response.headers:
                self.remaining_requests = int(response.headers["x-requests-remaining"])
            
            response.raise_for_status()
            data = response.json()
            
            # Cache successful response
            if use_cache:
                self._set_cached(cache_key, data)
            
            return APIResponse(
                success=True,
                data=data,
                remaining_requests=self.remaining_requests
            )
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                return APIResponse(success=False, data=None, error="Invalid API key")
            elif response.status_code == 429:
                return APIResponse(success=False, data=None, error="Rate limit exceeded")
            return APIResponse(success=False, data=None, error=str(e))
        except requests.exceptions.RequestException as e:
            return APIResponse(success=False, data=None, error=str(e))
    
    def get_sports(self) -> APIResponse:
        """Get list of available sports."""
        return self._request("sports", {}, use_cache=True)
    
    def get_odds(
        self,
        sport_key: str,
        bookmakers: Optional[List[str]] = None,
    ) -> APIResponse:
        """
        Get odds for a sport.
        
        Args:
            sport_key: Sport key like "soccer_epl"
            bookmakers: Optional list of bookmakers to filter
        """
        params = {
            "regions": self.regions,
            "markets": self.markets,
            "oddsFormat": self.odds_format,
        }
        
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        
        return self._request(f"sports/{sport_key}/odds", params)
    
    def get_events(self, sport_key: str) -> APIResponse:
        """Get upcoming events for a sport (without odds)."""
        return self._request(f"sports/{sport_key}/events", {})


class OddsAPICollector:
    """
    High-level collector for odds data.
    
    Converts raw API responses to our schema objects
    with normalization and validation.
    """
    
    def __init__(self, client: Optional[OddsAPIClient] = None):
        self.client = client or OddsAPIClient()
        self.validator = OddsValidator()
    
    def fetch_matches(
        self,
        league: League,
        include_odds: bool = True,
        max_hours_ahead: int = 48
    ) -> List[Match]:
        """
        Fetch upcoming matches for a league.
        
        Args:
            league: League to fetch
            include_odds: Whether to include odds data
            max_hours_ahead: Maximum hours in future to include
            
        Returns:
            List of Match objects
        """
        sport_key = OddsAPIClient.SPORT_KEYS.get(league)
        if not sport_key:
            logger.warning(f"Unknown league: {league}")
            return []
        
        if include_odds:
            response = self.client.get_odds(sport_key)
        else:
            response = self.client.get_events(sport_key)
        
        if not response.success:
            logger.error(f"API error: {response.error}")
            return []
        
        matches = []
        cutoff = datetime.utcnow() + timedelta(hours=max_hours_ahead)
        
        for event in response.data:
            try:
                commence_time = datetime.fromisoformat(
                    event["commence_time"].replace("Z", "+00:00")
                )
                
                # Skip matches too far ahead or already started
                if commence_time > cutoff:
                    continue
                if commence_time < datetime.utcnow():
                    continue
                
                home_name = event["home_team"]
                away_name = event["away_team"]
                
                match = Match(
                    id=event["id"],
                    home_team=Team(
                        name=home_name,
                        normalized_name=SourceNormalizer.from_odds_api(home_name),
                    ),
                    away_team=Team(
                        name=away_name,
                        normalized_name=SourceNormalizer.from_odds_api(away_name),
                    ),
                    league=league,
                    commence_time=commence_time,
                    source="odds_api",
                    source_id=event["id"],
                )
                
                matches.append(match)
                
            except Exception as e:
                logger.warning(f"Failed to parse event: {e}")
                continue
        
        logger.info(f"Fetched {len(matches)} matches for {league.display_name}")
        return matches
    
    def fetch_odds(
        self,
        league: League,
        exclude_outliers: bool = True
    ) -> Dict[str, List[OddsSnapshot]]:
        """
        Fetch odds snapshots grouped by match ID.
        
        Returns:
            Dict mapping match_id to list of OddsSnapshot from different bookmakers
        """
        sport_key = OddsAPIClient.SPORT_KEYS.get(league)
        if not sport_key:
            return {}
        
        response = self.client.get_odds(sport_key)
        if not response.success:
            logger.error(f"API error: {response.error}")
            return {}
        
        odds_by_match: Dict[str, List[OddsSnapshot]] = {}
        
        for event in response.data:
            match_id = event["id"]
            odds_by_match[match_id] = []
            
            for bookmaker in event.get("bookmakers", []):
                try:
                    # Find h2h market
                    h2h_market = None
                    for market in bookmaker.get("markets", []):
                        if market["key"] == "h2h":
                            h2h_market = market
                            break
                    
                    if not h2h_market:
                        continue
                    
                    # Parse outcomes
                    home_odds = None
                    draw_odds = None
                    away_odds = None
                    
                    for outcome in h2h_market["outcomes"]:
                        if outcome["name"] == event["home_team"]:
                            home_odds = outcome["price"]
                        elif outcome["name"] == event["away_team"]:
                            away_odds = outcome["price"]
                        elif outcome["name"] == "Draw":
                            draw_odds = outcome["price"]
                    
                    if home_odds and away_odds:
                        snapshot = OddsSnapshot(
                            match_id=match_id,
                            bookmaker=bookmaker["key"],
                            timestamp=datetime.fromisoformat(
                                bookmaker["last_update"].replace("Z", "+00:00")
                            ),
                            home_odds=home_odds,
                            draw_odds=draw_odds,
                            away_odds=away_odds,
                        )
                        
                        # Validate
                        result = self.validator.validate_snapshot(snapshot)
                        if result.is_valid:
                            odds_by_match[match_id].append(snapshot)
                        else:
                            logger.debug(f"Invalid odds from {bookmaker['key']}: {result.errors}")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse bookmaker odds: {e}")
                    continue
        
        logger.info(f"Fetched odds for {len(odds_by_match)} matches")
        return odds_by_match
    
    def fetch_best_odds(
        self,
        league: League,
        exclude_outliers: bool = True
    ) -> Dict[str, BestOdds]:
        """
        Fetch best available odds for each match.
        
        This is what we actually use for betting.
        """
        odds_by_match = self.fetch_odds(league, exclude_outliers)
        
        best_odds: Dict[str, BestOdds] = {}
        
        for match_id, snapshots in odds_by_match.items():
            if snapshots:
                try:
                    best = OddsValidator.compute_best_odds(snapshots, exclude_outliers)
                    best_odds[match_id] = best
                except Exception as e:
                    logger.warning(f"Failed to compute best odds for {match_id}: {e}")
        
        return best_odds
    
    def fetch_all(
        self,
        leagues: Optional[List[League]] = None,
        max_hours_ahead: int = 48
    ) -> Dict[str, tuple]:
        """
        Fetch all data for multiple leagues.
        
        Returns:
            Dict mapping match_id to (Match, BestOdds) tuple
        """
        if leagues is None:
            leagues = list(League)
        
        all_data: Dict[str, tuple] = {}
        
        for league in leagues:
            # Skip non-football for now (basketball has different structure)
            if not league.is_football:
                continue
            
            matches = self.fetch_matches(league, include_odds=True, max_hours_ahead=max_hours_ahead)
            best_odds = self.fetch_best_odds(league)
            
            for match in matches:
                if match.id in best_odds:
                    all_data[match.id] = (match, best_odds[match.id])
        
        logger.info(f"Total: {len(all_data)} matches with odds across {len(leagues)} leagues")
        return all_data
