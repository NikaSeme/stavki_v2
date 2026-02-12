"""
Team name normalization processor.

Critical for matching teams across different data sources.
Each source uses different naming conventions.

The normalizer:
1. Lowercases and strips whitespace
2. Removes common suffixes (FC, United, etc.)
3. Maps known aliases to canonical names
4. Handles unicode and special characters
5. Fuzzy matches for unknown teams
"""

import re
from typing import Dict, Optional, Tuple
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


# Canonical team aliases - THE source of truth
# Keys are normalized forms, values are canonical names
TEAM_ALIASES: Dict[str, str] = {
    # England - Premier League
    "man utd": "manchester united",
    "man united": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "manchester c": "manchester city",
    "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur",
    "tottenham hotspur fc": "tottenham hotspur",
    "wolves": "wolverhampton wanderers",
    "wolverhampton": "wolverhampton wanderers",
    "west ham": "west ham united",
    "west ham utd": "west ham united",
    "brighton": "brighton and hove albion",
    "brighton hove albion": "brighton and hove albion",
    "nottingham": "nottingham forest",
    "nottm forest": "nottingham forest",
    "nott'm forest": "nottingham forest",
    "newcastle": "newcastle united",
    "newcastle utd": "newcastle united",
    "leicester": "leicester city",
    "sheffield utd": "sheffield united",
    "sheffield": "sheffield united",
    "luton": "luton town",
    "burnley fc": "burnley",
    "everton fc": "everton",
    "arsenal fc": "arsenal",
    "chelsea fc": "chelsea",
    "liverpool fc": "liverpool",
    "brentford fc": "brentford",
    "fulham fc": "fulham",
    "crystal palace fc": "crystal palace",
    "bournemouth": "afc bournemouth",
    "afc bournemouth": "bournemouth",
    
    # England - Championship
    "leeds": "leeds united",
    "hull": "hull city",
    "ipswich": "ipswich town",
    "coventry": "coventry city",
    "bristol": "bristol city",
    "plymouth": "plymouth argyle",
    "qpr": "queens park rangers",
    "swansea": "swansea city",
    "cardiff": "cardiff city",
    "norwich": "norwich city",
    "watford fc": "watford",
    "burnley fc": "burnley",
    "wba": "west bromwich albion",
    "west brom": "west bromwich albion",
    "birmingham": "birmingham city",
    "sheff wed": "sheffield wednesday",
    "sheffield wed": "sheffield wednesday",
    "stoke": "stoke city",
    "preston": "preston north end",
    "preston ne": "preston north end",
    "millwall fc": "millwall",
    "huddersfield": "huddersfield town",
    "blackburn": "blackburn rovers",
    "sunderland afc": "sunderland",
    "middlesbrough fc": "middlesbrough",
    "rotherham": "rotherham united",
    
    # Spain - La Liga
    "atletico": "atletico madrid",
    "atl madrid": "atletico madrid",
    "atlético madrid": "atletico madrid",
    "real madrid cf": "real madrid",
    "fc barcelona": "barcelona",
    "barca": "barcelona",
    "athletic": "athletic bilbao",
    "athletic club": "athletic bilbao",
    "real sociedad": "real sociedad",
    "celta": "celta vigo",
    "celta de vigo": "celta vigo",
    "real betis": "betis",
    "betis sevilla": "betis",
    "sevilla fc": "sevilla",
    "villarreal cf": "villarreal",
    "valencia cf": "valencia",
    "rayo": "rayo vallecano",
    "getafe cf": "getafe",
    "osasuna": "ca osasuna",
    "ca osasuna": "osasuna",
    "almeria": "ud almeria",
    "ud almeria": "almeria",
    "cadiz cf": "cadiz",
    "mallorca": "rcd mallorca",
    "rcd mallorca": "mallorca",
    "girona fc": "girona",
    "alaves": "deportivo alaves",
    "deportivo alaves": "alaves",
    "las palmas": "ud las palmas",
    "ud las palmas": "las palmas",
    
    # Germany - Bundesliga
    "bayern": "bayern munich",
    "bayern munchen": "bayern munich",
    "fc bayern": "bayern munich",
    "bayern münchen": "bayern munich",
    "dortmund": "borussia dortmund",
    "borussia dortmund": "borussia dortmund",
    "bvb": "borussia dortmund",
    "rb leipzig": "rb leipzig",
    "leipzig": "rb leipzig",
    "leverkusen": "bayer leverkusen",
    "bayer 04 leverkusen": "bayer leverkusen",
    "gladbach": "borussia monchengladbach",
    "borussia m'gladbach": "borussia monchengladbach",
    "monchengladbach": "borussia monchengladbach",
    "mönchengladbach": "borussia monchengladbach",
    "frankfurt": "eintracht frankfurt",
    "sge": "eintracht frankfurt",
    "wolfsburg": "vfl wolfsburg",
    "vfl wolfsburg": "wolfsburg",
    "freiburg": "sc freiburg",
    "sc freiburg": "freiburg",
    "hoffenheim": "tsg hoffenheim",
    "tsg hoffenheim": "hoffenheim",
    "koln": "fc koln",
    "köln": "fc koln",
    "fc cologne": "fc koln",
    "cologne": "fc koln",
    "mainz": "mainz 05",
    "1. fsv mainz 05": "mainz 05",
    "augsburg": "fc augsburg",
    "fc augsburg": "augsburg",
    "union berlin": "union berlin",
    "1. fc union berlin": "union berlin",
    "hertha": "hertha berlin",
    "hertha bsc": "hertha berlin",
    "werder bremen": "werder bremen",
    "sv werder bremen": "werder bremen",
    "stuttgart": "vfb stuttgart",
    "vfb stuttgart": "stuttgart",
    "bochum": "vfl bochum",
    "vfl bochum": "bochum",
    "heidenheim": "1. fc heidenheim",
    "1. fc heidenheim": "heidenheim",
    "darmstadt": "sv darmstadt 98",
    "sv darmstadt 98": "darmstadt",
    
    # Italy - Serie A
    "inter": "inter milan",
    "internazionale": "inter milan",
    "fc internazionale": "inter milan",
    "inter milano": "inter milan",
    "ac milan": "milan",
    "milan ac": "milan",
    "juve": "juventus",
    "juventus fc": "juventus",
    "napoli": "napoli",
    "ssc napoli": "napoli",
    "roma": "as roma",
    "as roma": "roma",
    "lazio": "lazio",
    "ss lazio": "lazio",
    "atalanta": "atalanta",
    "atalanta bc": "atalanta",
    "fiorentina": "fiorentina",
    "acf fiorentina": "fiorentina",
    "torino": "torino",
    "torino fc": "torino",
    "bologna": "bologna",
    "bologna fc": "bologna",
    "udinese": "udinese",
    "udinese calcio": "udinese",
    "sassuolo": "sassuolo",
    "us sassuolo": "sassuolo",
    "verona": "hellas verona",
    "hellas verona fc": "hellas verona",
    "lecce": "lecce",
    "us lecce": "lecce",
    "empoli": "empoli",
    "empoli fc": "empoli",
    "monza": "monza",
    "ac monza": "monza",
    "cagliari": "cagliari",
    "cagliari calcio": "cagliari",
    "genoa": "genoa",
    "genoa cfc": "genoa",
    "salernitana": "salernitana",
    "us salernitana": "salernitana",
    "frosinone": "frosinone",
    "frosinone calcio": "frosinone",
    
    # France - Ligue 1
    "psg": "paris saint-germain",
    "paris sg": "paris saint-germain",
    "paris saint germain": "paris saint-germain",
    "paris": "paris saint-germain",
    "marseille": "olympique marseille",
    "om": "olympique marseille",
    "olympique de marseille": "olympique marseille",
    "lyon": "olympique lyon",
    "ol": "olympique lyon",
    "olympique lyonnais": "olympique lyon",
    "monaco": "as monaco",
    "as monaco": "monaco",
    "lille": "lille osc",
    "lille osc": "lille",
    "losc lille": "lille",
    "rennes": "stade rennais",
    "stade rennais fc": "stade rennais",
    "nice": "ogc nice",
    "ogc nice": "nice",
    "lens": "rc lens",
    "rc lens": "lens",
    "nantes": "fc nantes",
    "fc nantes": "nantes",
    "reims": "stade de reims",
    "stade reims": "stade de reims",
    "montpellier": "montpellier hsc",
    "montpellier hsc": "montpellier",
    "strasbourg": "rc strasbourg",
    "rc strasbourg": "strasbourg",
    "brest": "stade brestois",
    "stade brestois 29": "stade brestois",
    "toulouse": "toulouse fc",
    "toulouse fc": "toulouse",
    "clermont": "clermont foot",
    "clermont foot 63": "clermont foot",
    "lorient": "fc lorient",
    "fc lorient": "lorient",
    "metz": "fc metz",
    "fc metz": "metz",
    "le havre": "le havre ac",
    "le havre ac": "le havre",
}

# Suffixes to remove during normalization
REMOVE_SUFFIXES = [
    " fc", " cf", " sc", " ac", " ssc", " bc", " afc",
    " united", " city", " town", " rovers", " wanderers",
    " hotspur", " albion", " athletic", " villa",
]


def _basic_normalize(name: str) -> str:
    """Basic string normalization."""
    # Lowercase and strip
    name = name.lower().strip()
    
    # Replace common unicode chars
    replacements = {
        'ü': 'u', 'ö': 'o', 'ä': 'a', 'ß': 'ss',
        'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
        'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a',
        'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o',
        'ú': 'u', 'ù': 'u', 'û': 'u',
        'ñ': 'n', 'ç': 'c', 'í': 'i', 'ì': 'i', 'î': 'i',
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    
    # Remove special characters except spaces and hyphens
    name = re.sub(r'[^\w\s\-]', '', name)
    
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name)
    
    return name.strip()


@lru_cache(maxsize=1000)
def normalize_team_name(name: str, remove_suffix: bool = False) -> str:
    """
    Normalize a team name to canonical form.
    
    Args:
        name: Raw team name from data source
        remove_suffix: If True, remove FC/United/etc suffixes
        
    Returns:
        Normalized canonical team name
    """
    if not name:
        return ""
    
    # Basic normalization
    normalized = _basic_normalize(name)
    
    # Check aliases first (exact match after basic normalization)
    if normalized in TEAM_ALIASES:
        return TEAM_ALIASES[normalized]
    
    # Try with suffix removal
    if remove_suffix:
        for suffix in REMOVE_SUFFIXES:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
                break
    
    # Check aliases again after suffix removal
    if normalized in TEAM_ALIASES:
        return TEAM_ALIASES[normalized]
    
    # Return as-is if no alias found
    return normalized


def get_canonical_name(name: str) -> Tuple[str, bool]:
    """
    Get canonical team name with flag indicating if alias was used.
    
    Returns:
        (canonical_name, was_aliased)
    """
    normalized = normalize_team_name(name)
    basic = _basic_normalize(name)
    
    return normalized, (normalized != basic)


def add_team_alias(alias: str, canonical: str) -> None:
    """
    Add a new team alias at runtime.
    
    Useful for handling unknown teams encountered dynamically.
    """
    normalized_alias = _basic_normalize(alias)
    TEAM_ALIASES[normalized_alias] = canonical
    # Clear cache since aliases changed
    normalize_team_name.cache_clear()
    logger.info(f"Added team alias: {alias} -> {canonical}")


def suggest_match(unknown: str, threshold: float = 0.8) -> Optional[str]:
    """
    Suggest a possible canonical name for an unknown team using fuzzy matching.
    
    Args:
        unknown: Unknown team name
        threshold: Minimum similarity score (0-1)
        
    Returns:
        Suggested canonical name or None
    """
    try:
        from difflib import SequenceMatcher
    except ImportError:
        return None
    
    normalized_unknown = _basic_normalize(unknown)
    
    best_match = None
    best_score = 0
    
    # Get all unique canonical names
    canonicals = set(TEAM_ALIASES.values())
    
    for canonical in canonicals:
        score = SequenceMatcher(None, normalized_unknown, canonical).ratio()
        if score > best_score:
            best_score = score
            best_match = canonical
    
    if best_score >= threshold:
        return best_match
    
    return None


# Source-specific normalizers
class SourceNormalizer:
    """
    Source-specific team name normalization.
    
    Each data source has its own naming conventions.
    """
    
    ODDS_API_OVERRIDES: Dict[str, str] = {
        # Odds API sometimes uses full names
        "Manchester City FC": "manchester city",
        "Manchester United FC": "manchester united",
    }
    
    SPORTMONKS_OVERRIDES: Dict[str, str] = {
        # SportMonks variations
        "Nott'ham Forest": "nottingham forest",
    }
    
    @classmethod
    def from_odds_api(cls, name: str) -> str:
        """Normalize team name from Odds API."""
        if name in cls.ODDS_API_OVERRIDES:
            return cls.ODDS_API_OVERRIDES[name]
        return normalize_team_name(name)
    
    @classmethod
    def from_sportmonks(cls, name: str) -> str:
        """Normalize team name from SportMonks."""
        if name in cls.SPORTMONKS_OVERRIDES:
            return cls.SPORTMONKS_OVERRIDES[name]
        return normalize_team_name(name)
    
    @classmethod
    def from_football_data_uk(cls, name: str) -> str:
        """Normalize team name from football-data.co.uk."""
        return normalize_team_name(name)
