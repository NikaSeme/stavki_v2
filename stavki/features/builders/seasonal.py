"""
Seasonal / Time-of-Season Feature Builder (Tier 3).

Captures temporal context that affects match outcomes:
- season_progress: 0.0–1.0 based on position within Aug–May season
- is_end_of_season: flag for last ~6 matchdays (increased motivation/pressure)
- matchday_normalized: matchday / total matchdays
- is_weekend, is_evening_kickoff: scheduling context
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# Typical European football season runs Aug–May (~10 months)
SEASON_START_MONTH = 8   # August
SEASON_END_MONTH = 5     # May
SEASON_LENGTH_DAYS = 300  # ~10 months


class SeasonalFeatureBuilder:
    """
    Compute temporal/seasonal context features for a match.

    No historical fitting needed — purely derived from match datetime
    and optional matchday metadata.
    """

    name = "seasonal"

    def __init__(self, total_matchdays: int = 38):
        self.total_matchdays = total_matchdays
        self._is_fitted = False

    def fit(self, matches: List[Match]) -> None:
        """No fitting required for seasonal features."""
        self._is_fitted = True
        logger.info("SeasonalFeatureBuilder: ready (no fitting needed)")

    def _season_progress(self, dt: datetime) -> float:
        """
        Compute season progress 0.0–1.0.

        Uses month-based heuristic:
          Aug=0.0, Sep=0.05, ..., Dec=0.4, Jan=0.5, ..., May=1.0
        """
        month = dt.month
        if month >= SEASON_START_MONTH:
            # Aug(8)=0, Sep(9)=1, ..., Dec(12)=4
            months_elapsed = month - SEASON_START_MONTH
        else:
            # Jan(1)=5, Feb(2)=6, ..., May(5)=9
            months_elapsed = (12 - SEASON_START_MONTH) + month

        # Total months in season = 10 (Aug through May)
        total_months = 10
        progress = months_elapsed / total_months

        # Refine with day-of-month
        progress += (dt.day / 30.0) / total_months

        return round(min(max(progress, 0.0), 1.0), 3)

    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get seasonal features for a match."""
        dt = None
        if match:
            dt = match.commence_time
        elif as_of:
            dt = as_of

        if not dt:
            return {
                "season_progress": 0.5,
                "is_end_of_season": 0.0,
                "matchday_normalized": 0.5,
                "is_weekend": 0.0,
                "is_evening_kickoff": 0.0,
                "month_sin": 0.0,
                "month_cos": 1.0,
            }

        import math

        progress = self._season_progress(dt)
        is_weekend = 1.0 if dt.weekday() >= 5 else 0.0  # Sat=5, Sun=6
        is_evening = 1.0 if dt.hour >= 17 else 0.0  # 5pm+

        # Matchday from metadata (if available)
        matchday = match.matchday if match and match.matchday else None
        if matchday:
            md_norm = round(matchday / self.total_matchdays, 3)
        else:
            md_norm = progress  # Use season progress as proxy

        # End of season flag (last 15% of season = ~6 matchdays)
        is_end = 1.0 if progress > 0.85 else 0.0

        # Cyclical month encoding (captures periodicity)
        month_angle = 2 * math.pi * (dt.month - 1) / 12
        month_sin = round(math.sin(month_angle), 3)
        month_cos = round(math.cos(month_angle), 3)

        return {
            "season_progress": progress,
            "is_end_of_season": is_end,
            "matchday_normalized": md_norm,
            "is_weekend": is_weekend,
            "is_evening_kickoff": is_evening,
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
