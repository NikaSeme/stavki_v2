"""
Betfair Exchange API Client and Collector.

Requires:
- App Key
- Username/Password
- Client Certificate (.crt) and Key (.key) for non-interactive login.

Documentation: https://docs.developer.betfair.com/
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Betfair Constants
AUTH_URL = "https://identitysso.betfair.com/api/certlogin"
API_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"

@dataclass
class BetfairConfig:
    app_key: str
    username: str
    password: str
    cert_file: str
    key_file: str

class BetfairClient:
    """
    Client for Betfair Exchange API (JSON-RPC).
    Handles authentication and session management.
    """
    
    # Competition IDs
    COMPETITIONS = {
        "EPL": "47",
        "LA_LIGA": "117",
        "BUNDESLIGA": "59",
        "SERIE_A": "81",
        "LIGUE_1": "55",
        "CHAMPIONSHIP": "2022802", # EFL Championship
    }

    def __init__(self, config: Optional[BetfairConfig] = None):
        self.config = config or self._load_config()
        self.session_token = None
        self.session = requests.Session()
        
    def _load_config(self) -> BetfairConfig:
        """Load config from environment variables."""
        return BetfairConfig(
            app_key=os.getenv("BETFAIR_APP_KEY", ""),
            username=os.getenv("BETFAIR_USERNAME", ""),
            password=os.getenv("BETFAIR_PASSWORD", ""),
            cert_file=os.getenv("BETFAIR_CERT_FILE", "client-2048.crt"),
            key_file=os.getenv("BETFAIR_KEY_FILE", "client-2048.key"),
        )

    def login(self) -> bool:
        """Perform cert-based login."""
        if not self.config.app_key:
            logger.warning("Betfair App Key not set.")
            return False
            
        if not os.path.exists(self.config.cert_file) or not os.path.exists(self.config.key_file):
            logger.warning(f"Betfair cert/key files not found: {self.config.cert_file}, {self.config.key_file}")
            return False

        try:
            payload = f"username={self.config.username}&password={self.config.password}"
            headers = {
                "X-Application": self.config.app_key,
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            resp = requests.post(
                AUTH_URL,
                data=payload,
                cert=(self.config.cert_file, self.config.key_file),
                headers=headers,
                timeout=10
            )
            
            data = resp.json()
            if data.get("loginStatus") == "SUCCESS":
                self.session_token = data["sessionToken"]
                logger.info("Betfair login successful")
                return True
            else:
                logger.error(f"Betfair login failed: {data.get('loginStatus')}")
                return False
                
        except Exception as e:
            logger.error(f"Betfair login exception: {e}")
            return False

    def request(self, method: str, params: Dict) -> Dict:
        """Make JSON-RPC request."""
        if not self.session_token:
            if not self.login():
                 return {"error": "Login failed"}
        
        headers = {
            "X-Application": self.config.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        json_rpc = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        try:
            resp = self.session.post(API_URL, json=json_rpc, headers=headers, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            
            if "error" in result:
                # Handle session expiry logic if needed
                logger.error(f"Betfair API Error: {result['error']}")
                return {}
                
            return result.get("result", {})
            
        except Exception as e:
            logger.error(f"Betfair request failed: {e}")
            return {}

    def get_market_catalogue(
        self, 
        competition_ids: List[str], 
        max_results: int = 100
    ) -> List[Dict]:
        """Get 'Match Odds' markets for competitions."""
        return self.request(
            "SportsAPING/v1.0/listMarketCatalogue",
            {
                "filter": {
                    "competitionIds": competition_ids,
                    "marketTypeCodes": ["MATCH_ODDS"],
                    "marketBettingTypes": ["ODDS"],
                    "turnInPlayEnabled": True # Also get upcoming
                },
                "maxResults": max_results,
                "marketProjection": ["RUNNER_METADATA", "EVENT", "MARKET_START_TIME"]
            }
        )

    def get_market_book(self, market_ids: List[str]) -> List[Dict]:
        """Get prices for markets."""
        return self.request(
            "SportsAPING/v1.0/listMarketBook",
            {
                "marketIds": market_ids,
                "priceProjection": {
                    "priceData": ["EX_BEST_OFFERS"],
                    "exBestOffersOverrides": {"bestPricesDepth": 1}
                }
            }
        )


class BetfairCollector:
    """
    High-level collector for Betfair Exchange data.
    """
    
    def __init__(self, client: Optional[BetfairClient] = None):
        self.client = client or BetfairClient()
        from ..schemas import League
        from ..processors.validate import OddsValidator
        self.validator = OddsValidator()
        self.League = League # Helper alias

    def fetch_odds(
        self, 
        league: "League", 
        matches: Optional[List["Match"]] = None
    ) -> Dict[str, List["OddsSnapshot"]]:
        """
        Fetch Betfair Exchange (Red/Blue) prices.
        
        Strategy:
        1. Fetch all Markets for the league.
        2. Match Markets to our 'matches' list by team names (fuzzy matching).
        3. Fetch Prices for matched markets.
        4. Return snapshots.
        """
        from ..schemas import OddsSnapshot, League
        from ..processors.normalize import SourceNormalizer
        from difflib import SequenceMatcher

        # Map League to Betfair Competition ID
        comp_id = None
        if league == League.EPL: comp_id = BetfairClient.COMPETITIONS["EPL"]
        elif league == League.LA_LIGA: comp_id = BetfairClient.COMPETITIONS["LA_LIGA"]
        elif league == League.BUNDESLIGA: comp_id = BetfairClient.COMPETITIONS["BUNDESLIGA"]
        elif league == League.SERIE_A: comp_id = BetfairClient.COMPETITIONS["SERIE_A"]
        elif league == League.LIGUE_1: comp_id = BetfairClient.COMPETITIONS["LIGUE_1"]
        elif league == League.CHAMPIONSHIP: comp_id = BetfairClient.COMPETITIONS["CHAMPIONSHIP"]
        
        if not comp_id:
            logger.warning(f"Betfair Comp ID not found for {league}")
            return {}
            
        # 1. Get Markets
        markets = self.client.get_market_catalogue([comp_id])
        if not markets:
            return {}
            
        # 2. Match Markets to our Matches
        # If matches not provided, we can't key by match_id easily without creating them first.
        # Assuming matches provided (from SportMonks pipeline)
        if not matches:
            logger.warning("BetfairCollector needs 'matches' list to map odds correctly.")
            return {}
            
        market_map = {} # market_id -> match_id
        
        # Simple name matching
        # Betfair Event Name: "Man Utd v Liverpool"
        for mkt in markets:
            event = mkt.get("event", {})
            event_name = event.get("name", "")
            market_id = mkt.get("marketId")
            
            # Find best match in our matches list
            best_score = 0
            best_match = None
            
            for m in matches:
                # fast check: date
                # mkt["marketStartTime"] "2024-02-15T20:00:00.000Z"
                # Check if dates are close (within 24h)
                # Skip tricky date parsing for now, rely on names
                
                # Compare "Home v Away"
                our_str = f"{m.home_team.name} v {m.away_team.name}"
                ratio = SequenceMatcher(None, event_name, our_str).ratio()
                if ratio > 0.6 and ratio > best_score:
                    best_score = ratio
                    best_match = m
            
            if best_match and best_score > 0.8: # Threshold
                market_map[market_id] = best_match
        
        if not market_map:
            logger.info("No Betfair markets matched to internal matches.")
            return {}
            
        # 3. Get Prices
        market_ids = list(market_map.keys())
        # Bulk fetch (max 40 usually per req, assume <40 for now or chunk loop)
        books = self.client.get_market_book(market_ids[:40]) 
        
        results = {}
        
        for book in books:
            market_id = book.get("marketId")
            match = market_map.get(market_id)
            if not match: continue
            
            runners = book.get("runners", [])
            
            # Map runners to Home/Away/Draw
            # Need runner metadata from catalogue to know which is which?
            # get_market_catalogue returns runners metadata in marketProjection.
            # I need to store that mapping from step 1.
            
            # Re-find catalogue entry for this market
            cat_entry = next((m for m in markets if m["marketId"] == market_id), None)
            if not cat_entry: continue
            
            runner_meta = cat_entry.get("runners", []) # [{selectionId, runnerName, ...}]
            
            home_price = None
            draw_price = None
            away_price = None
            
            for r in runners:
                selection_id = r.get("selectionId")
                # Find name
                search_name = next((rm["runnerName"] for rm in runner_meta if rm["selectionId"] == selection_id), "")
                
                # Get Best Back Price
                available_to_back = r.get("ex", {}).get("availableToBack", [])
                best_price = available_to_back[0]["price"] if available_to_back else None
                
                if not best_price: continue
                
                # Identify runner
                # "The Draw"
                if "Draw" in search_name:
                    draw_price = best_price
                    continue
                
                # Check normalized names
                norm_name = SourceNormalizer.from_football_data_uk(search_name)
                # Compare with match home/away normalized
                # Assuming SourceNormalizer handles "Man Utd" -> "Manchester United" etc.
                # Use fuzzy match again if needed?
                
                # Simple containment
                if match.home_team.normalized_name in norm_name or norm_name in match.home_team.normalized_name:
                    home_price = best_price
                elif match.away_team.normalized_name in norm_name or norm_name in match.away_team.normalized_name:
                    away_price = best_price
            
            if home_price and draw_price and away_price:
                 snap = OddsSnapshot(
                    match_id=match.id,
                    bookmaker="Betfair Exchange",
                    timestamp=datetime.now(), # Realtime
                    home_odds=home_price,
                    draw_odds=draw_price,
                    away_odds=away_price
                 )
                 
                 results[match.id] = [snap]
                 
        return results
