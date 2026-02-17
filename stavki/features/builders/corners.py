"""
Corners Feature Builder (Tier 3).

Computes rolling corner statistics from match data.
Previously wired (collected from CSV HC/AC columns) but unused in models.

Features:
- rolling_corners_for_home/away: average corners per game
- rolling_corners_against_home/away: average corners conceded
- corners_diff: expected corner differential (home vs away)
- corners_total: combined expected total (for over/under corners market)
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)


class CornersFeatureBuilder:
    """
    Compute rolling corners statistics per team from historical match data.

    Uses match.stats.corners_home / corners_away which are populated
    from CSV columns HC / AC.
    """

    name = "corners"

    def __init__(self, rolling_window: int = 10):
        self.rolling_window = rolling_window
        # team -> list of {corners_for, corners_against, date}
        self._team_corners: Dict[str, list] = defaultdict(list)
        self._global_avg = 0.0
        self._is_fitted = False

    def fit(self, matches: List[Match]) -> None:
        """Build rolling corners history from historical matches."""
        self._team_corners.clear()
        all_corners = []

        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.stats or not m.is_completed:
                continue

            ch = m.stats.corners_home
            ca = m.stats.corners_away

            if ch is None and ca is None:
                continue

            ch = ch or 0
            ca = ca or 0

            home = m.home_team.normalized_name
            away = m.away_team.normalized_name

            self._team_corners[home].append({
                "for": ch,
                "against": ca,
                "date": m.commence_time,
            })
            self._team_corners[away].append({
                "for": ca,
                "against": ch,
                "date": m.commence_time,
            })

            all_corners.extend([ch, ca])

            # Trim histories to avoid unbounded growth
            for team in [home, away]:
                if len(self._team_corners[team]) > self.rolling_window * 3:
                    self._team_corners[team] = \
                        self._team_corners[team][-self.rolling_window * 3:]

        self._global_avg = sum(all_corners) / len(all_corners) if all_corners else 5.0
        self._is_fitted = True

        logger.info(
            f"CornersFeatureBuilder: {len(self._team_corners)} teams profiled | "
            f"global avg corners per side = {self._global_avg:.1f}"
        )

    def _get_rolling(
        self, team: str, ref_time: Optional[datetime]
    ) -> tuple:
        """Get rolling corners for/against for a team. Returns (avg_for, avg_against)."""
        history = self._team_corners.get(team, [])
        if ref_time:
            history = [h for h in history if h["date"] < ref_time]
        recent = history[-self.rolling_window:]

        if not recent:
            return self._global_avg, self._global_avg

        avg_for = sum(h["for"] for h in recent) / len(recent)
        avg_against = sum(h["against"] for h in recent) / len(recent)
        return avg_for, avg_against

    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get corners features for a match."""
        defaults = {
            "rolling_corners_for_home": round(self._global_avg, 2),
            "rolling_corners_for_away": round(self._global_avg, 2),
            "rolling_corners_against_home": round(self._global_avg, 2),
            "rolling_corners_against_away": round(self._global_avg, 2),
            "corners_diff": 0.0,
            "corners_total": round(2 * self._global_avg, 2),
        }

        if not match:
            return defaults

        ref_time = as_of or match.commence_time

        home_for, home_against = self._get_rolling(
            match.home_team.normalized_name, ref_time
        )
        away_for, away_against = self._get_rolling(
            match.away_team.normalized_name, ref_time
        )

        features = {
            "rolling_corners_for_home": round(home_for, 2),
            "rolling_corners_for_away": round(away_for, 2),
            "rolling_corners_against_home": round(home_against, 2),
            "rolling_corners_against_away": round(away_against, 2),
            "corners_diff": round(home_for - away_for, 2),
            "corners_total": round(home_for + away_for, 2),
        }

        return features
