"""
Historical data loader.

Loads historical match data from football-data.co.uk CSV files.
Used for backtesting and model training.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Iterator
import logging

from ..schemas import Match, MatchResult, Team, League, Outcome

logger = logging.getLogger(__name__)


# football-data.co.uk column mappings
COLUMN_MAPPINGS = {
    "date": ["Date"],
    "home": ["HomeTeam", "Home", "HT"],
    "away": ["AwayTeam", "Away", "AT"],
    "fthg": ["FTHG", "HG"],  # Full time home goals
    "ftag": ["FTAG", "AG"],  # Full time away goals
    "ftr": ["FTR", "Res"],   # Full time result (H/D/A)
    "b365h": ["B365H", "BbMxH"],  # Bet365 home odds
    "b365d": ["B365D", "BbMxD"],  # Bet365 draw odds
    "b365a": ["B365A", "BbMxA"],  # Bet365 away odds
    "psh": ["PSH", "PH"],    # Pinnacle home
    "psd": ["PSD", "PD"],    # Pinnacle draw
    "psa": ["PSA", "PA"],    # Pinnacle away
}

# League identifiers from football-data.co.uk
FOOTBALL_DATA_LEAGUES = {
    "E0": League.EPL,            # English Premier League
    "E1": League.CHAMPIONSHIP,   # English Championship
    "SP1": League.LA_LIGA,       # Spanish La Liga
    "D1": League.BUNDESLIGA,     # German Bundesliga
    "I1": League.SERIE_A,        # Italian Serie A
    "F1": League.LIGUE_1,        # French Ligue 1
}


class FootballDataLoader:
    """
    Loader for football-data.co.uk historical data.
    
    Data source: https://www.football-data.co.uk/data.php
    
    Expected file format: CSV with columns like Date, HomeTeam, AwayTeam, FTHG, FTAG, etc.
    """
    
    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
    
    def _get_column(self, row: Dict, key: str) -> Optional[str]:
        """Get value from row trying multiple column name variants."""
        for col_name in COLUMN_MAPPINGS.get(key, [key]):
            if col_name in row:
                return row[col_name]
        return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date in various formats."""
        formats = [
            "%d/%m/%Y",
            "%d/%m/%y",
            "%Y-%m-%d",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def _parse_odds(self, value: Optional[str]) -> Optional[float]:
        """Parse odds value."""
        if not value:
            return None
        try:
            odds = float(value)
            if odds >= 1.0:
                return odds
        except ValueError:
            pass
        return None
    
    def load_file(
        self,
        file_path: Path,
        league: Optional[League] = None,
        season: Optional[str] = None
    ) -> Iterator[Match]:
        """
        Load matches from a single CSV file.
        
        Yields Match objects with results.
        """
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                
                for i, row in enumerate(reader):
                    try:
                        date_str = self._get_column(row, "date")
                        if not date_str:
                            continue
                        
                        match_date = self._parse_date(date_str)
                        if not match_date:
                            continue
                        
                        home_team = self._get_column(row, "home")
                        away_team = self._get_column(row, "away")
                        
                        if not home_team or not away_team:
                            continue
                        
                        # Parse score
                        fthg = self._get_column(row, "fthg")
                        ftag = self._get_column(row, "ftag")
                        
                        home_score = int(fthg) if fthg and fthg.isdigit() else None
                        away_score = int(ftag) if ftag and ftag.isdigit() else None
                        
                        # Determine league if not provided
                        if league is None:
                            # Try to infer from file path
                            for code, lg in FOOTBALL_DATA_LEAGUES.items():
                                if code in file_path.stem:
                                    league = lg
                                    break
                            if league is None:
                                league = League.EPL  # Default
                        
                        # Generate match ID
                        match_id = f"fd_{league.value}_{match_date.strftime('%Y%m%d')}_{home_team}_{away_team}"
                        match_id = match_id.replace(" ", "_").lower()
                        
                        match = Match(
                            id=match_id,
                            home_team=Team(name=home_team),
                            away_team=Team(name=away_team),
                            league=league,
                            commence_time=match_date,
                            home_score=home_score,
                            away_score=away_score,
                            season=season,
                            source="football_data_uk",
                        )
                        
                        yield match
                        
                    except Exception as e:
                        logger.debug(f"Error parsing row {i}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
    
    def load_season(
        self,
        league_code: str,
        season: str
    ) -> List[Match]:
        """
        Load a season of data.
        
        Args:
            league_code: football-data.co.uk league code (E0, SP1, etc.)
            season: Season in format "2324" for 2023-24
            
        Returns:
            List of Match objects
        """
        # Try common file naming patterns
        possible_files = [
            self.data_dir / f"{league_code}_{season}.csv",
            self.data_dir / league_code / f"season-{season}.csv",
            self.data_dir / f"{season}" / f"{league_code}.csv",
        ]
        
        league = FOOTBALL_DATA_LEAGUES.get(league_code, League.EPL)
        
        for file_path in possible_files:
            if file_path.exists():
                return list(self.load_file(file_path, league, season))
        
        logger.warning(f"No data file found for {league_code} {season}")
        return []
    
    def load_all_seasons(
        self,
        league_code: str,
        start_season: str = "1920",
        end_season: str = "2526"
    ) -> List[Match]:
        """Load multiple seasons of data for a league."""
        all_matches = []
        
        # Generate season codes (1920, 2021, 2122, etc.)
        start_year = int(start_season[:2])
        end_year = int(end_season[:2])
        
        for year in range(start_year, end_year + 1):
            season = f"{year:02d}{(year+1) % 100:02d}"
            matches = self.load_season(league_code, season)
            all_matches.extend(matches)
            if matches:
                logger.info(f"Loaded {len(matches)} matches for {league_code} {season}")
        
        return all_matches
    
    def load_consolidated_csv(
        self,
        file_path: Path,
        league: Optional[League] = None
    ) -> List[Match]:
        """
        Load a consolidated multi-season CSV file.
        
        For pre-combined datasets.
        """
        return list(self.load_file(file_path, league))


class HistoricalOddsExtractor:
    """
    Extract historical odds from football-data.co.uk files.
    
    The files include opening odds from various bookmakers.
    """
    
    BOOKMAKER_COLUMNS = {
        "bet365": ("B365H", "B365D", "B365A"),
        "pinnacle": ("PSH", "PSD", "PSA"),
        "betway": ("BWH", "BWD", "BWA"),
        "interwetten": ("IWH", "IWD", "IWA"),
        "williamhill": ("WHH", "WHD", "WHA"),
        "vc": ("VCH", "VCD", "VCA"),
        "market_max": ("BbMxH", "BbMxD", "BbMxA"),
        "market_avg": ("BbAvH", "BbAvD", "BbAvA"),
    }
    
    @classmethod
    def extract_odds_from_row(
        cls,
        row: Dict,
        bookmaker: str = "bet365"
    ) -> Optional[Dict[str, float]]:
        """Extract odds from a CSV row for a specific bookmaker."""
        cols = cls.BOOKMAKER_COLUMNS.get(bookmaker)
        if not cols:
            return None
        
        home_col, draw_col, away_col = cols
        
        try:
            home = float(row.get(home_col, 0))
            draw = float(row.get(draw_col, 0))
            away = float(row.get(away_col, 0))
            
            if home > 1 and draw > 1 and away > 1:
                return {
                    "home": home,
                    "draw": draw,
                    "away": away,
                }
        except (ValueError, TypeError):
            pass
        
        return None
    
    @classmethod
    def get_best_odds_from_row(cls, row: Dict) -> Optional[Dict[str, float]]:
        """Get best odds across all bookmakers."""
        best_home = 0.0
        best_draw = 0.0
        best_away = 0.0
        
        for bookmaker in cls.BOOKMAKER_COLUMNS.keys():
            odds = cls.extract_odds_from_row(row, bookmaker)
            if odds:
                best_home = max(best_home, odds["home"])
                best_draw = max(best_draw, odds["draw"])
                best_away = max(best_away, odds["away"])
        
        if best_home > 1 and best_draw > 1 and best_away > 1:
            return {
                "home": best_home,
                "draw": best_draw,
                "away": best_away,
            }
        
        return None
