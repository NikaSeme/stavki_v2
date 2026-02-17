"""
Advanced statistics feature builder.

Computes rolling averages for:
- Expected Goals (xG) For/Against — uses synthetic xG from shot data when
  official xG is unavailable (which is the common case in CSV data)
- Shots / Shots on Target
- xG Performance (Goals - xG)

Performance: Pre-indexes matches by team during initialization to avoid
O(N²) full-scan per team per match.
"""

from typing import List, Optional, Dict
from datetime import datetime
from collections import defaultdict
from bisect import bisect_left
import pandas as pd
import numpy as np
import logging

from stavki.features import FeatureBuilder
from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# Synthetic xG coefficients (same as SyntheticXGBuilder defaults)
# Used when official xG is not available
SYNTH_XG_COEFS = {
    "shots": 0.03,
    "sot": 0.12,
    "intercept": 0.05,
}


def _compute_synth_xg(shots: int, sot: int) -> float:
    """Compute synthetic xG from shot/SOT data when official xG is unavailable."""
    xg = (SYNTH_XG_COEFS["shots"] * (shots or 0) +
          SYNTH_XG_COEFS["sot"] * (sot or 0) +
          SYNTH_XG_COEFS["intercept"])
    return max(0.0, round(xg, 3))


class AdvancedFeatureBuilder(FeatureBuilder):
    """
    Computes rolling advanced statistics with pre-indexed team lookups.

    Performance improvement: O(N) per team instead of O(N²) total.
    """
    name = "advanced"

    def __init__(self, window: int = 5):
        self.window = window
        # Pre-indexed: team -> list of (commence_time, stats_dict)
        self._team_index: Dict[str, list] = defaultdict(list)
        self._is_fitted = False

    def fit(self, matches: List[Match]) -> None:
        """Pre-index matches by team for O(1) lookup instead of scanning."""
        self._team_index.clear()

        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.is_completed or not m.stats:
                continue

            for side, team in [
                ("home", m.home_team.normalized_name),
                ("away", m.away_team.normalized_name),
            ]:
                is_home = side == "home"

                # Get xG: prefer official, fall back to synthetic from shot data
                xg_for_val = (m.stats.xg_home if is_home else m.stats.xg_away)
                xg_against_val = (m.stats.xg_away if is_home else m.stats.xg_home)

                # If official xG is None/0, compute from shots
                if not xg_for_val:
                    shots_f = (m.stats.shots_home if is_home else m.stats.shots_away) or 0
                    sot_f = (m.stats.shots_on_target_home if is_home else m.stats.shots_on_target_away) or 0
                    xg_for_val = _compute_synth_xg(shots_f, sot_f)

                if not xg_against_val:
                    shots_a = (m.stats.shots_away if is_home else m.stats.shots_home) or 0
                    sot_a = (m.stats.shots_on_target_away if is_home else m.stats.shots_on_target_home) or 0
                    xg_against_val = _compute_synth_xg(shots_a, sot_a)

                record = {
                    "time": m.commence_time,
                    "xg_for": xg_for_val or 0.0,
                    "xg_against": xg_against_val or 0.0,
                    "shots_for": (m.stats.shots_home if is_home else m.stats.shots_away) or 0,
                    "shots_against": (m.stats.shots_away if is_home else m.stats.shots_home) or 0,
                    "sot_for": (m.stats.shots_on_target_home if is_home else m.stats.shots_on_target_away) or 0,
                    "sot_against": (m.stats.shots_on_target_away if is_home else m.stats.shots_on_target_home) or 0,
                    "goals": (m.home_score if is_home else m.away_score) or 0,
                }
                self._team_index[team].append(record)

        self._is_fitted = True
        logger.info(f"AdvancedFeatureBuilder: pre-indexed {len(self._team_index)} teams")

    def _get_recent_stats(
        self, team: str, as_of: Optional[datetime] = None
    ) -> list:
        """Get recent stats for team before as_of using pre-index."""
        records = self._team_index.get(team, [])
        if as_of:
            # Binary search for cutoff (records are sorted by time)
            # Find the rightmost index where time < as_of
            cutoff = bisect_left(
                [r["time"] for r in records], as_of
            )
            records = records[:cutoff]
        return records[-self.window:]

    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Get advanced stats features for a match."""
        if not match:
            return {}
            
        home = match.home_team.normalized_name
        away = match.away_team.normalized_name
        time = as_of or match.commence_time
        
        home_stats = self.compute_for_team(home, as_of=time)
        away_stats = self.compute_for_team(away, as_of=time)
        
        features = {}
        for k, v in home_stats.items():
            features[f"home_{k}"] = v
        for k, v in away_stats.items():
            features[f"away_{k}"] = v
            
        return features

    def compute_for_team(
        self,
        team: str,
        matches: List[Match] = None,  # Ignored when pre-indexed
        as_of: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Compute rolling stats for a team using pre-indexed data."""
        recent = self._get_recent_stats(team, as_of)

        if not recent:
            return {
                "xg_for": 0.0, "xg_against": 0.0,
                "shots_for": 0.0, "shots_against": 0.0,
                "sot_for": 0.0, "sot_against": 0.0,
                "xg_perf": 0.0
            }

        n = len(recent)
        avg_xg_for = sum(r["xg_for"] for r in recent) / n
        avg_xg_against = sum(r["xg_against"] for r in recent) / n
        avg_goals = sum(r["goals"] for r in recent) / n

        return {
            "xg_for": round(avg_xg_for, 3),
            "xg_against": round(avg_xg_against, 3),
            "xg_diff": round(avg_xg_for - avg_xg_against, 3),
            "shots_for": round(sum(r["shots_for"] for r in recent) / n, 2),
            "shots_against": round(sum(r["shots_against"] for r in recent) / n, 2),
            "sot_for": round(sum(r["sot_for"] for r in recent) / n, 2),
            "sot_against": round(sum(r["sot_against"] for r in recent) / n, 2),
            "xg_perf": round(avg_goals - avg_xg_for, 3),
        }

    def build(
        self,
        matches: List[Match],
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Build features for all matches."""
        # Auto-fit if not already fitted
        if not self._is_fitted:
            self.fit(matches)

        target_matches = matches
        if as_of:
            target_matches = [m for m in matches if m.commence_time < as_of]

        if not target_matches:
            return pd.DataFrame()

        rows = []
        for m in target_matches:
            home_stats = self.compute_for_team(
                m.home_team.normalized_name, as_of=m.commence_time
            )
            away_stats = self.compute_for_team(
                m.away_team.normalized_name, as_of=m.commence_time
            )

            row = {"match_id": m.id}
            for k, v in home_stats.items():
                row[f"home_{k}"] = v
            for k, v in away_stats.items():
                row[f"away_{k}"] = v
            rows.append(row)

        return pd.DataFrame(rows)
