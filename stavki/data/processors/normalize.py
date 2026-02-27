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
import difflib
from typing import Dict, Optional, Tuple
from functools import lru_cache
import logging
import csv
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MAPPING_DIR = DATA_DIR / "mapping"

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

class TeamMapper:
    """
    Handles team name mapping using CSV configuration.
    Source of Truth: data/mapping/canonical_teams.csv
    """
    
    _instance = None
    
    def __init__(self):
        self.canonical_teams = set()
        self.source_mappings: Dict[str, Dict[str, str]] = {} # source_name -> {raw_name -> canonical_name}
        self.aliases: Dict[str, str] = {} # normalized_alias -> canonical_name
        self._load_data()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
        
    def _load_data(self):
        # 1. Load Canonical Teams
        canon_path = MAPPING_DIR / "canonical_teams.csv"
        if canon_path.exists():
            with open(canon_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.canonical_teams.add(row["canonical_name"])
            logger.info(f"Loaded {len(self.canonical_teams)} canonical teams")
        else:
            logger.warning("canonical_teams.csv not found!")

        # 2. Load Source Mappings
        sources_dir = MAPPING_DIR / "sources"
        if sources_dir.exists():
            for csv_file in sources_dir.glob("*.csv"):
                source_name = csv_file.stem # e.g. "sportmonks"
                mapping = {}
                with open(csv_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        raw = row["raw_name"]
                        canon = row["canonical_name"]
                        if canon: # Only map if target exists
                            mapping[raw] = canon
                            # Also add to general aliases (normalized)
                            self.aliases[_basic_normalize(raw)] = canon
                self.source_mappings[source_name] = mapping
                logger.info(f"Loaded {len(mapping)} mappings for {source_name}")

        # 3. Load team_synonyms.json (central alias registry)
        import json
        synonyms_path = PROJECT_ROOT / "config" / "team_synonyms.json"
        if synonyms_path.exists():
            try:
                with open(synonyms_path, "r") as f:
                    synonyms = json.load(f)
                count = 0
                for raw_name, canonical in synonyms.items():
                    if raw_name.startswith("_"):
                        continue  # skip metadata keys like _comment
                    self.aliases[_basic_normalize(raw_name)] = canonical
                    count += 1
                logger.info(f"Loaded {count} aliases from team_synonyms.json")
            except Exception as e:
                logger.warning(f"Failed to load team_synonyms.json: {e}")

    def map_name(self, name: str, source: Optional[str] = None) -> Optional[str]:
        """Map a raw name to canonical name."""
        if not name:
            return None
            
        # 1. Exact canonical match?
        if name in self.canonical_teams:
            return name
            
        # 2. Source-specific mapping?
        if source and source in self.source_mappings:
            if name in self.source_mappings[source]:
                return self.source_mappings[source][name]
                
        # 3. Normalized alias match?
        norm = _basic_normalize(name)
        if norm in self.aliases:
            return self.aliases[norm]
            
        # 4. Global alias match (legacy support until fully migrated)
        # (We populate aliases from source maps, so this covers cross-source)
        
        return None

# Global Mapper Instance (lazy-loaded)
_mapper_instance = None

def _get_mapper():
    """Lazy-load TeamMapper to avoid file I/O at import time."""
    global _mapper_instance
    if _mapper_instance is None:
        _mapper_instance = TeamMapper.get_instance()
    return _mapper_instance

# Negative match cache to avoid repeated expensive fuzzy scans
_negative_cache: set = set()

# Legacy aliases dict — populated at runtime by add_team_alias()
TEAM_ALIASES: Dict[str, str] = {}


# Suffixes to remove during normalization
REMOVE_SUFFIXES = [
    " fc", " cf", " sc", " ac", " ssc", " bc", " afc",
    " united", " city", " town", " rovers", " wanderers",
    " hotspur", " albion", " athletic", " villa",
]




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
    
    # Check Mapper
    mapper = _get_mapper()
    mapped = mapper.map_name(name)
    if mapped:
        return mapped
        
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
    # Skip fuzzy match if this name already failed before
    if normalized in _negative_cache:
        return normalized
    
    auto_alias = auto_alias_high_confidence(normalized)
    if auto_alias:
        return auto_alias
    
    # Remember this name as unfamiliar to avoid future scans
    _negative_cache.add(normalized)
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
    SequenceMatcher = difflib.SequenceMatcher
    
    normalized_unknown = _basic_normalize(unknown)
    
    best_match = None
    best_score = 0
    
    # Get all unique canonical names
    canonicals = _get_mapper().canonical_teams

    
    for canonical in canonicals:
        score = SequenceMatcher(None, normalized_unknown, canonical).ratio()
        if score > best_score:
            best_score = score
            best_match = canonical
    
    if best_score >= threshold:
        return best_match
    
    return None


def auto_alias_high_confidence(unknown: str, threshold: float = 0.90) -> Optional[str]:
    """
    Attempt to automatically alias a team name if a very high confidence match is found.
    
    Args:
        unknown: Unknown team name
        threshold: strict threshold for auto-aliasing (default 0.90)
        
    Returns:
        Canonical name if matched and aliased, else None
    """
    suggestion = suggest_match(unknown, threshold=threshold)
    if suggestion:
        logger.info(f"Auto-aliasing high confidence match: '{unknown}' -> '{suggestion}'")
        add_team_alias(unknown, suggestion)
        return suggestion
    return None


# Source-specific normalizers
class SourceNormalizer:
    """
    Source-specific team name normalization.
    
    Each data source has its own naming conventions.
    """

    
    @classmethod
    def from_odds_api(cls, name: str) -> str:
        """Normalize team name from Odds API."""
        mapped = _get_mapper().map_name(name, source="odds_api")
        if mapped:
             return mapped
        return normalize_team_name(name)

    
    @classmethod
    def from_sportmonks(cls, name: str) -> str:
        """Normalize team name from SportMonks."""
        mapped = _get_mapper().map_name(name, source="sportmonks")
        if mapped:
             return mapped
        return normalize_team_name(name)

    
    @classmethod
    def from_football_data_uk(cls, name: str) -> str:
        """Normalize team name from football-data.co.uk."""
        return normalize_team_name(name)
