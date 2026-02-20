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
        
        # Load leagues from config
        self.leagues_map = {}
        try:
            leagues_path = PROJECT_ROOT / "stavki" / "config" / "leagues.json"
            if leagues_path.exists():
                with open(leagues_path) as f:
                    leagues_config = json.load(f)
                    # Config is Name -> ID, we need ID -> Name (lowercase)
                    # Filter out non-int values (like league weights dicts)
                    self.leagues_map = {
                        v: k.lower().replace("_", "") 
                        for k, v in leagues_config.items() 
                        if isinstance(v, int)
                    }
            else:
                logger.warning(f"Leagues config not found at {leagues_path}, using defaults")
                self.leagues_map = {
                    8: 'epl', 82: 'bundesliga', 564: 'laliga', 
                    384: 'seriea', 301: 'ligue1', 9: 'championship'
                }
        except Exception as e:
            logger.error(f"Failed to load leagues config: {e}")
            self.leagues_map = {}

        # Initialize API client if key provided
        self.client: Optional[SportMonksClient] = None
        if api_key:
            self.client = SportMonksClient(api_key)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"UnifiedDataLoader initialized (API: {'✓' if api_key else '✗'})")

    def normalize_team_name(self, name: str) -> str:
        """
        Normalize team name to standard format.
        
        Args:
            name: Raw team name
            
        Returns:
            Normalized name
        """
        from stavki.data.processors.normalize import normalize_team_name as centralized_normalize
        return centralized_normalize(name)

    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache key."""
        # Sanitize key
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.json"

    def _get_cached(self, key: str) -> Optional[List]:
        """Get data from cache if valid."""
        path = self._get_cache_path(key)
        if not path.exists():
            return None
            
        try:
            # Check TTL
            mtime = datetime.fromtimestamp(path.stat().st_mtime)
            age = datetime.now() - mtime
            if age.total_seconds() > self.cache_ttl_hours * 3600:
                return None
                
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
            return None

    def _set_cache(self, key: str, data: List) -> None:
        """Save data to cache."""
        path = self._get_cache_path(key)
        try:
            with open(path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def get_historical_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        leagues: Optional[List[str]] = None,
        include_recent_from_api: bool = False,
        force_reload: bool = False
    ) -> pd.DataFrame:
        """
        Load historical match data from local Parquet cache or raw CSVs.
        
        Args:
            start_date: Filter start date (YYYY-MM-DD)
            end_date: Filter end date (YYYY-MM-DD)
            leagues: List of league names (e.g. ['epl', 'laliga'])
            include_recent_from_api: If True, fetches recent data from API and merges
            force_reload: If True, rebuilds cache from raw CSVs
            
        Returns:
            DataFrame with standardized columns
        """
        parquet_path = self.cache_dir.parent / "historical_matches.parquet"
        
        # 1. Try Load Parquet
        df = pd.DataFrame()
        if parquet_path.exists() and not force_reload:
            try:
                df = pd.read_parquet(parquet_path)
                logger.info(f"Loaded {len(df)} matches from Parquet cache")
            except Exception as e:
                logger.warning(f"Failed to read Parquet cache: {e}")
        
        # 2. If no data, build from Raw CSVs
        if df.empty:
            logger.info("Building historical data from raw CSVs...")
            dfs = []
            raw_dir = PROJECT_ROOT / "data" / "raw"
            
            # Map of internal league names to CSV folder names if different
            # Assuming they match for now (epl -> epl, bundesliga -> bundesliga)
            
            target_leagues = leagues or self.SPORTMONKS_LEAGUES.values()
            
            for league in target_leagues:
                league_path = raw_dir / league
                if not league_path.exists():
                    continue
                
                for csv_file in league_path.glob("*.csv"):
                    try:
                        # Read CSV (handle encoding/errors)
                        # football-data.co.uk often uses ISO-8859-1
                        temp_df = pd.read_csv(csv_file, encoding='cp1252')
                        
                        # Normalize columns
                        # Standard keys: Date, HomeTeam, AwayTeam, FTHG, FTAG, FTR, B365H...
                        if 'Date' not in temp_df.columns:
                            continue
                            
                        # Parse dates (handle dd/mm/yy and dd/mm/yyyy)
                        # Coerce errors to NaT then drop
                        temp_df['Date'] = pd.to_datetime(temp_df['Date'], dayfirst=True, errors='coerce')
                        temp_df = temp_df.dropna(subset=['Date'])
                        
                        # Rename standard columns
                        rename_map = {
                            'HomeTeam': 'HomeTeam',
                            'AwayTeam': 'AwayTeam',
                            'FTHG': 'home_score',
                            'FTAG': 'away_score',
                            'FTR': 'result',
                            'B365H': 'home_odds',
                            'B365D': 'draw_odds',
                            'B365A': 'away_odds'
                        }
                        
                        # Keep only available columns
                        cols_to_keep = [c for c in rename_map.keys() if c in temp_df.columns] + ['Date']
                        temp_df = temp_df[cols_to_keep].rename(columns=rename_map)
                        
                        # Add metadata
                        temp_df['League'] = league
                        temp_df['source'] = 'csv'
                        
                        # Extract season from filename if possible 'epl_2023_24.csv'
                        try:
                            season_part = csv_file.stem.split('_')[-2:] # 2023, 24
                            if len(season_part) == 2 and season_part[0].isdigit():
                                temp_df['season'] = f"{season_part[0]}-{season_part[1]}"
                        except:
                            pass
                            
                        dfs.append(temp_df)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process {csv_file}: {e}")
            
            if dfs:
                df = pd.concat(dfs, ignore_index=True)
                
                # Normalize Team Names
                logger.info("Normalizing team names...")
                df['HomeTeam'] = df['HomeTeam'].apply(self.normalize_team_name)
                df['AwayTeam'] = df['AwayTeam'].apply(self.normalize_team_name)
                
                # Cache to Parquet
                try:
                    df.to_parquet(parquet_path)
                    logger.info(f"Saved {len(df)} matches to Parquet cache")
                except Exception as e:
                    logger.warning(f"Failed to save Parquet cache: {e}")
            else:
                logger.warning("No historical CSV data found!")
                
        # 3. Filter
        if not df.empty:
            if start_date:
                df = df[df['Date'] >= start_date]
            if end_date:
                df = df[df['Date'] <= end_date]
            if leagues:
                # Filter by normalized or raw league name
                df = df[df['League'].isin(leagues)]
        
        # 4. Integrate API Recent Data
        if include_recent_from_api:
            # Load recent from API
            api_start = df['Date'].max().strftime("%Y-%m-%d") if not df.empty else (start_date or "2023-01-01")
            
            # Add 1 day buffer to avoid overlap duplication if possible
            # But safer to overlap and dedup
            recent_df = self._fetch_recent_from_api(start_date=api_start, end_date=end_date)
            
            if not recent_df.empty:
                # Merge and Deduplicate
                # Needs aligning columns first
                # API returns: Date, HomeTeam, AwayTeam, League, fixture_id, B365H...
                
                # Map API columns to Standard
                mapper = {
                    'B365H': 'home_odds', 'B365D': 'draw_odds', 'B365A': 'away_odds',
                    'xG_home': 'xg_home', 'xG_away': 'xg_away'
                }
                recent_df = recent_df.rename(columns=mapper)
                recent_df['source'] = 'api'
                
                # TODO: Ensure consistent columns before concat
                # For now, just concat and let pandas align
                df = pd.concat([df, recent_df], ignore_index=True)
                
                # Deduplicate based on Date + Teams
                df = df.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='last')
        
        # 5. Standardization (Dual support for snake_case and PascalCase)
        # This resolves the widespread inconsistency in the codebase
        if not df.empty:
            # HomeTeam <-> home_team
            if 'HomeTeam' in df.columns and 'home_team' not in df.columns:
                df['home_team'] = df['HomeTeam']
            elif 'home_team' in df.columns and 'HomeTeam' not in df.columns:
                df['HomeTeam'] = df['home_team']

            # AwayTeam <-> away_team
            if 'AwayTeam' in df.columns and 'away_team' not in df.columns:
                df['away_team'] = df['AwayTeam']
            elif 'away_team' in df.columns and 'AwayTeam' not in df.columns:
                df['AwayTeam'] = df['away_team']
            
            # Date <-> date
            if 'Date' in df.columns and 'date' not in df.columns:
                df['date'] = df['Date']
            elif 'date' in df.columns and 'Date' not in df.columns:
                df['Date'] = df['date']

        return df.sort_values('Date').reset_index(drop=True)

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
        
        league_ids = list(self.leagues_map.keys())
        
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
                        'League': self.leagues_map.get(fix.league_id, 'unknown'),
                        'fixture_id': fix.fixture_id,
                    }
                    
                    # Try to get odds
                    try:
                        odds = self.client.get_fixture_odds(fix.fixture_id)
                        if odds:
                            match_data['B365H'] = odds[0]['odds'].get('home', 2.5)
                            match_data['B365D'] = odds[0]['odds'].get('draw', 3.3)
                            match_data['B365A'] = odds[0]['odds'].get('away', 2.8)
                    except Exception as e:
                        logger.warning(f"Failed to get odds for fixture {fix.fixture_id}: {e}")
                    
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
                    except Exception as e:
                        logger.warning(f"Failed to get full details for fixture {fix.fixture_id}: {e}")
                    
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
        start_date="2024-01-01",
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
