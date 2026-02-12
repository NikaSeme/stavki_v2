"""
Venue Feature Builder (Tier 2).

Features based on venue properties:
- Capacity (normalized) → larger venues = more home advantage
- Surface type (artificial vs grass) → affects play style
- Altitude factor → high altitude = more goals
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# Average EPL stadium capacity for normalization
AVG_CAPACITY = 35000
MAX_CAPACITY = 80000


class VenueFeatureBuilder:
    """
    Compute venue-based features.
    
    Caches venue info from enrichment data across matches.
    """
    
    name = "venue"
    
    def __init__(self):
        # venue_name -> { capacity, surface, altitude }
        self._venue_cache: Dict[str, Dict] = {}
        # team -> home_venue
        self._team_venue: Dict[str, str] = {}
        self._is_fitted = False
    
    def fit(self, matches: List[Match]) -> None:
        """Cache venue info from historical matches."""
        self._venue_cache.clear()
        self._team_venue.clear()
        
        for m in matches:
            if m.enrichment and m.enrichment.venue_info:
                vi = m.enrichment.venue_info
                if vi.name:
                    self._venue_cache[vi.name.lower()] = {
                        "capacity": vi.capacity or AVG_CAPACITY,
                        "surface": vi.surface or "grass",
                        "altitude": vi.altitude_m or 0,
                    }
                    self._team_venue[m.home_team.normalized_name] = vi.name.lower()
            elif m.venue:
                venue_key = m.venue.lower()
                if venue_key not in self._venue_cache:
                    self._venue_cache[venue_key] = {
                        "capacity": AVG_CAPACITY,
                        "surface": "grass",
                        "altitude": 0,
                    }
                self._team_venue[m.home_team.normalized_name] = venue_key
        
        self._is_fitted = True
        logger.info(f"VenueFeatureBuilder: {len(self._venue_cache)} venues cached")
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get venue features for a match."""
        defaults = {
            "venue_capacity_norm": 0.5,
            "is_artificial_pitch": 0.0,
            "altitude_factor": 0.0,
        }
        
        if not match:
            return defaults
        
        # Try to get venue info
        venue_data = None
        
        # From enrichment
        if match.enrichment and match.enrichment.venue_info:
            vi = match.enrichment.venue_info
            venue_data = {
                "capacity": vi.capacity or AVG_CAPACITY,
                "surface": vi.surface or "grass",
                "altitude": vi.altitude_m or 0,
            }
        # From cache
        elif match.venue and match.venue.lower() in self._venue_cache:
            venue_data = self._venue_cache[match.venue.lower()]
        # From team's known home venue
        elif match.home_team.normalized_name in self._team_venue:
            vkey = self._team_venue[match.home_team.normalized_name]
            venue_data = self._venue_cache.get(vkey)
        
        if not venue_data:
            return defaults
        
        capacity = venue_data.get("capacity", AVG_CAPACITY)
        surface = venue_data.get("surface", "grass")
        altitude = venue_data.get("altitude", 0)
        
        return {
            "venue_capacity_norm": round(
                min(capacity / MAX_CAPACITY, 1.0), 3
            ),
            "is_artificial_pitch": 1.0 if surface.lower() in ("artificial", "synthetic", "astroturf") else 0.0,
            "altitude_factor": round(
                min(altitude / 2000, 1.0), 3  # Normalize: 2000m = max effect
            ),
        }
