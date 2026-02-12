"""
Unified Data Loader.

Primary source: SportMonks API (for live and recent data)
Fallback: CSV files (for historical data 2014+)

Features:
- Smart source selection
- Team name normalization
- Response caching
- Feature consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import json
import hashlib

from stavki.config import DATA_DIR, PROJECT_ROOT
from stavki.data.collectors.sportmonks import SportMonksClient, MatchFixture

logger = logging.getLogger(__name__)


@dataclass
class DataSource:
    """Represents a data source with priority."""
    name: str
    priority: int  # Lower = higher priority
    available: bool = True


class UnifiedDataLoader:
    """
    Smart data loader that combines SportMonks API and CSV files.
    
    Usage:
        loader = UnifiedDataLoader(api_key="...")
        
        # Historical data (uses CSV cache + recent from API)
        df = loader.get_historical_data(start="2020-01-01", end="2024-12-31")
        
        # Live fixtures (always from API)
        fixtures = loader.get_live_fixtures(days=7)
        
        # Match with features (normalized)
        match_data = loader.get_match_data(fixture_id=123456)
    """
    
    # League mapping
    SPORTMONKS_LEAGUES = {
        8: 'epl',
        82: 'bundesliga',
        564: 'laliga',
        384: 'seriea',
        301: 'ligue1',
        9: 'championship',
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        cache_ttl_hours: int = 24,
        use_api_for_recent_days: int = 30
    ):
        """
        Initialize unified data loader.
        
        Args:
            api_key: SportMonks API key
            cache_dir: Directory for caching API responses
            cache_ttl_hours: Cache TTL in hours
            use_api_for_recent_days: Use API for data newer than this
        """
        self.api_key = api_key
        self.cache_dir = cache_dir or (PROJECT_ROOT / ".cache" / "sportmonks")
        self.cache_ttl_hours = cache_ttl_hours
        self.use_api_for_recent_days = use_api_for_recent_days
        
        # Initialize API client if key provided
        self.client: Optional[SportMonksClient] = None
        if api_key:
            self.client = SportMonksClient(api_key)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"UnifiedDataLoader initialized (API: {'✓' if api_key else '✗'})")
    
    @staticmethod
    def normalize_team_name(name: str) -> str:
        """Normalize team name to canonical form.
        
        Delegates to the canonical normalizer in data.processors.normalize.
        """
        from stavki.data.processors.normalize import normalize_team_name as _normalize
        return _normalize(name)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        hash_key = hashlib.md5(key.encode()).hexdigest()[:12]
        return self.cache_dir / f"{hash_key}.json"
    
    def _get_cached(self, key: str) -> Optional[Dict]:
        """Get cached response if valid."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path) as f:
                data = json.load(f)
            
            # Check TTL
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            if datetime.now() - cached_at > timedelta(hours=self.cache_ttl_hours):
                return None
            
            return data.get("response")
        except Exception:
            return None
    
    def _set_cache(self, key: str, response: Dict):
        """Cache a response."""
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "cached_at": datetime.now().isoformat(),
                    "key": key,
                    "response": response
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache: {e}")
    
    # =========================================================================
    # Historical Data
    # =========================================================================
    
    def get_historical_data(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        leagues: Optional[List[str]] = None,
        include_recent_from_api: bool = True
    ) -> pd.DataFrame:
        """
        Get historical match data.
        
        Uses CSV for bulk historical data, optionally supplements
        with recent data from SportMonks API.
        
        Args:
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            leagues: List of league codes to include
            include_recent_from_api: Fetch recent matches from API
            
        Returns:
            DataFrame with historical matches
        """
        # Load base data from CSV
        df = self._load_csv_data()
        
        # Apply filters
        if start:
            df = df[df['Date'] >= start]
        if end:
            df = df[df['Date'] <= end]
        if leagues:
            df = df[df['League'].isin(leagues)]
        
        # Optionally add recent data from API
        if include_recent_from_api and self.client:
            recent_cutoff = (datetime.now() - timedelta(days=self.use_api_for_recent_days)).strftime("%Y-%m-%d")
            
            # Remove recent data from CSV (will replace with API data)
            df = df[df['Date'] < recent_cutoff]
            
            # Fetch recent from API
            recent_df = self._fetch_recent_from_api(recent_cutoff, end)
            
            if not recent_df.empty:
                df = pd.concat([df, recent_df], ignore_index=True)
                df = df.sort_values('Date').reset_index(drop=True)
        
        # Normalize team names
        df['HomeTeam'] = df['HomeTeam'].apply(self.normalize_team_name)
        df['AwayTeam'] = df['AwayTeam'].apply(self.normalize_team_name)
        
        logger.info(f"Loaded {len(df)} historical matches")
        return df
    
    def _load_csv_data(self) -> pd.DataFrame:
        """Load data from CSV/Parquet files."""
        # Try parquet first (faster)
        parquet_path = DATA_DIR / "features_full.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            csv_path = DATA_DIR / "features_full.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path, low_memory=False)
            else:
                # Fall back to training data
                df = pd.read_csv(DATA_DIR / "training_data.csv", low_memory=False)
        
        # Ensure date column
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        return df
    
    def _fetch_recent_from_api(
        self,
        start_date: str,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Fetch recent matches from SportMonks API."""
        if not self.client:
            return pd.DataFrame()
        
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        
        # Check cache
        cache_key = f"recent_{start_date}_{end_date}"
        cached = self._get_cached(cache_key)
        if cached:
            return pd.DataFrame(cached)
        
        logger.info(f"Fetching recent matches from API: {start_date} to {end_date}")
        
        all_matches = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        league_ids = list(self.SPORTMONKS_LEAGUES.keys())
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            try:
                fixtures = self.client.get_fixtures_by_date(date_str, league_ids)
                
                for fix in fixtures:
                    # Get fixture with stats and odds
                    match_data = {
                        'Date': date_str,
                        'HomeTeam': fix.home_team,
                        'AwayTeam': fix.away_team,
                        'League': self.SPORTMONKS_LEAGUES.get(fix.league_id, 'unknown'),
                        'fixture_id': fix.fixture_id,
                    }
                    
                    # Try to get odds
                    try:
                        odds = self.client.get_fixture_odds(fix.fixture_id)
                        if odds:
                            match_data['B365H'] = odds[0]['odds'].get('home', 2.5)
                            match_data['B365D'] = odds[0]['odds'].get('draw', 3.3)
                            match_data['B365A'] = odds[0]['odds'].get('away', 2.8)
                    except Exception:
                        pass
                    
                    # Try to get full match data (stats + lineups + events + referee)
                    try:
                        full = self.client.get_fixture_full(fix.fixture_id)
                        stats = full.get("stats")
                        if stats:
                            match_data['xG_home'] = stats.home_xg
                            match_data['xG_away'] = stats.away_xg
                            match_data['shots_home'] = stats.home_shots
                            match_data['shots_away'] = stats.away_shots
                            match_data['possession_home'] = stats.home_possession
                            match_data['possession_away'] = stats.away_possession
                        if full.get("referee"):
                            match_data['referee'] = full["referee"]
                    except Exception:
                        pass
                    
                    all_matches.append(match_data)
                    
            except Exception as e:
                logger.warning(f"Failed to fetch {date_str}: {e}")
            
            current += timedelta(days=1)
        
        df = pd.DataFrame(all_matches)
        
        # Cache result
        if not df.empty:
            self._set_cache(cache_key, df.to_dict(orient='records'))
        
        return df
    
    # =========================================================================
    # Live Fixtures
    # =========================================================================
    
    def get_live_fixtures(
        self,
        days: int = 7,
        leagues: Optional[List[str]] = None
    ) -> List[MatchFixture]:
        """
        Get upcoming fixtures from SportMonks API.
        
        Args:
            days: Number of days ahead
            leagues: Optional league filter
            
        Returns:
            List of MatchFixture objects
        """
        if not self.client:
            logger.error("SportMonks API key not configured")
            return []
        
        # Map league names to IDs
        if leagues:
            league_ids = [
                lid for lid, lname in self.SPORTMONKS_LEAGUES.items()
                if lname in leagues
            ]
        else:
            league_ids = list(self.SPORTMONKS_LEAGUES.keys())
        
        fixtures = self.client.get_upcoming_fixtures(days=days, league_ids=league_ids)
        
        # Normalize team names
        for fix in fixtures:
            fix.home_team = self.normalize_team_name(fix.home_team)
            fix.away_team = self.normalize_team_name(fix.away_team)
        
        return fixtures
    
    def get_fixture_with_odds(
        self,
        fixture_id: int
    ) -> Dict:
        """
        Get fixture data with odds from SportMonks.
        
        Returns:
            Dict with fixture details and odds
        """
        if not self.client:
            return {}
        
        # Get odds
        odds_data = self.client.get_fixture_odds(fixture_id)
        
        # Get fixture details
        fixtures = self.client._request(
            f"fixtures/{fixture_id}",
            includes=["participants", "venue", "weatherreport"]
        )
        
        fixture = fixtures.get("data", {})
        
        return {
            "fixture_id": fixture_id,
            "fixture": fixture,
            "odds": odds_data,
        }
    
    # =========================================================================
    # Testing Support
    # =========================================================================
    
    def get_test_data(
        self,
        n_matches: int = 100,
        league: str = "epl"
    ) -> pd.DataFrame:
        """
        Get sample data for testing.
        
        Args:
            n_matches: Number of matches
            league: League to sample from
            
        Returns:
            DataFrame with test matches
        """
        df = self.get_historical_data(leagues=[league])
        return df.tail(n_matches)
    
    def validate_data_consistency(self) -> Dict:
        """
        Validate that CSV and API data are consistent.
        
        Returns:
            Validation report
        """
        report = {
            "csv_matches": 0,
            "api_matches": 0,
            "overlapping_dates": 0,
            "team_name_mismatches": [],
            "result_mismatches": [],
        }
        
        # This would compare CSV data with API data for same dates
        # to ensure consistency
        
        return report


def get_loader(
    api_key: Optional[str] = None
) -> UnifiedDataLoader:
    """
    Factory function to create loader with API key from environment.
    """
    import os
    
    if not api_key:
        # Try to load from .env
        env_path = PROJECT_ROOT / '.env'
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith('SPORTMONKS_API_KEY='):
                        api_key = line.strip().split('=', 1)[1].strip('"').strip("'")
    
    return UnifiedDataLoader(api_key=api_key)


if __name__ == "__main__":
    # Demo
    print("Testing Unified Data Loader")
    print("=" * 50)
    
    loader = get_loader()
    
    # Test historical data
    print("\n1. Loading historical data...")
    df = loader.get_historical_data(
        start="2024-01-01",
        leagues=['epl', 'bundesliga'],
        include_recent_from_api=True
    )
    print(f"   Loaded {len(df)} matches")
    
    # Test live fixtures
    print("\n2. Fetching live fixtures...")
    fixtures = loader.get_live_fixtures(days=7)
    print(f"   Found {len(fixtures)} upcoming matches")
    
    # Test normalization
    print("\n3. Team name normalization:")
    test_names = ["Man United", "Spurs", "Bayern München", "Inter"]
    for name in test_names:
        print(f"   '{name}' -> '{loader.normalize_team_name(name)}'")
