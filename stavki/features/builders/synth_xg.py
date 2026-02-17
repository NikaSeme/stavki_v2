"""
Synthetic xG Feature Builder (Tier 1).

Estimates expected goals from per-player shot data when official xG
is not available on the API plan.

Two modes:
  1. Calibrated: fits coefficients from historical goals ~ f(shots, sot, big_chances)
     using ridge regression on the training data.
  2. Fallback: uses default coefficients when insufficient shot data exists.

Features produced:
  - synth_xg_home / synth_xg_away — per-match estimated xG
  - synth_xg_diff — home advantage differential
  - synth_xg_overperform_home / _away — actual goals vs expected (luck signal)
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import logging
import numpy as np

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# Regression coefficients mapping shot data → xG.
# Used as fallback when insufficient data to calibrate.
DEFAULT_COEFS = {
    "shots": 0.03,
    "sot": 0.12,
    "big_chances": 0.35,
    "intercept": 0.05,
}


class SyntheticXGBuilder:
    """
    Compute synthetic xG features from per-player shot data.

    During fit(), calibrates coefficients from historical goals using
    ridge regression. Falls back to DEFAULT_COEFS when data is insufficient.
    """

    name = "synth_xg"

    def __init__(self, rolling_window: int = 10, min_calibration_samples: int = 30):
        self.rolling_window = rolling_window
        self.min_calibration_samples = min_calibration_samples
        self.coefs = DEFAULT_COEFS.copy()
        self.calibrated = False
        # team -> list of recent { synth_xg, actual_goals }
        self._team_xg_history: Dict[str, list] = defaultdict(list)
        self._is_fitted = False

    def _compute_match_xg(self, shots: float, sot: float,
                          big_chances: float) -> float:
        """Compute synthetic xG from shot data using current coefficients."""
        xg = (self.coefs["shots"] * shots +
              self.coefs["sot"] * sot +
              self.coefs["big_chances"] * big_chances +
              self.coefs["intercept"])
        return max(0.0, round(xg, 3))

    def _extract_team_shots(self, match: Match, side: str) -> dict:
        """Extract shot data from match lineups or stats."""
        result = {"shots": 0, "sot": 0, "big_chances": 0}

        if not match.lineups:
            # Fall back to match stats
            if match.stats:
                if side == "home":
                    result["shots"] = match.stats.shots_home or 0
                    result["sot"] = match.stats.shots_on_target_home or 0
                else:
                    result["shots"] = match.stats.shots_away or 0
                    result["sot"] = match.stats.shots_on_target_away or 0
            return result

        # Get per-player data from lineups
        lineup = match.lineups.home if side == "home" else match.lineups.away
        if not lineup or not lineup.starting_xi:
            return result

        for p in lineup.starting_xi:
            # PlayerEntry might have extra fields if populated from enriched data
            player_dict = p.model_dump() if hasattr(p, 'model_dump') else {}
            result["shots"] += player_dict.get("shots", 0) or 0
            result["sot"] += player_dict.get("shots_on_target", 0) or 0
            bc = (player_dict.get("big_chances_created", 0) or 0) + \
                 (player_dict.get("big_chances_missed", 0) or 0)
            result["big_chances"] += bc

        # If no per-player shot data, fall back to match stats
        if result["shots"] == 0 and match.stats:
            if side == "home":
                result["shots"] = match.stats.shots_home or 0
                result["sot"] = match.stats.shots_on_target_home or 0
            else:
                result["shots"] = match.stats.shots_away or 0
                result["sot"] = match.stats.shots_on_target_away or 0

        return result

    def _calibrate_coefficients(self, matches: List[Match]) -> bool:
        """
        Calibrate xG coefficients from historical goals ~ f(shots, sot, big_chances).

        Uses ridge regression to fit on actual match data. This is the key
        improvement over hardcoded coefficients — the model now learns from
        your specific dataset.

        Returns True if calibration succeeded, False if insufficient data.
        """
        X_rows = []
        y_rows = []

        for m in matches:
            for side in ["home", "away"]:
                shot_data = self._extract_team_shots(m, side)

                # Only use matches with actual shot data (non-zero)
                if shot_data["shots"] == 0 and shot_data["sot"] == 0:
                    continue

                goals = (m.home_score if side == "home" else m.away_score) or 0

                X_rows.append([
                    shot_data["shots"],
                    shot_data["sot"],
                    shot_data["big_chances"],
                ])
                y_rows.append(goals)

        if len(X_rows) < self.min_calibration_samples:
            logger.info(
                f"SyntheticXGBuilder: Only {len(X_rows)} calibration samples "
                f"(need {self.min_calibration_samples}), using default coefficients"
            )
            return False

        X = np.array(X_rows, dtype=np.float64)
        y = np.array(y_rows, dtype=np.float64)

        # Ridge regression: goals ~ shots * β1 + sot * β2 + big_chances * β3 + intercept
        # Using closed-form solution: β = (X'X + λI)^(-1) X'y
        # λ = 1.0 for mild regularization (prevents overfitting on small data)
        n_features = X.shape[1]
        X_bias = np.column_stack([X, np.ones(len(X))])  # Add intercept column
        lam = 1.0
        I = np.eye(X_bias.shape[1])
        I[-1, -1] = 0  # Don't regularize intercept

        try:
            beta = np.linalg.solve(
                X_bias.T @ X_bias + lam * I,
                X_bias.T @ y
            )

            # Enforce non-negative coefficients (xG should increase with more shots)
            beta[:n_features] = np.maximum(beta[:n_features], 0.001)

            self.coefs = {
                "shots": round(float(beta[0]), 5),
                "sot": round(float(beta[1]), 5),
                "big_chances": round(float(beta[2]), 5),
                "intercept": round(float(beta[3]), 5),
            }

            # Compute R² for diagnostics
            y_pred = X_bias @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            logger.info(
                f"SyntheticXGBuilder: Calibrated on {len(X_rows)} samples | "
                f"R²={r_squared:.3f} | coefs: shots={self.coefs['shots']:.4f}, "
                f"sot={self.coefs['sot']:.4f}, bc={self.coefs['big_chances']:.4f}, "
                f"intercept={self.coefs['intercept']:.4f}"
            )
            return True

        except np.linalg.LinAlgError:
            logger.warning("SyntheticXGBuilder: Ridge regression failed, using defaults")
            return False

    def fit(self, matches: List[Match]) -> None:
        """Build rolling xG history per team from historical matches."""
        self._team_xg_history.clear()

        # Step 1: Calibrate coefficients from historical data
        sorted_matches = sorted(matches, key=lambda x: x.commence_time)
        self.calibrated = self._calibrate_coefficients(sorted_matches)

        # Step 2: Build rolling xG history using calibrated coefficients
        all_xg_home = []
        all_xg_away = []

        for m in sorted_matches:
            for side, team_name, goals in [
                ("home", m.home_team.normalized_name, m.home_score),
                ("away", m.away_team.normalized_name, m.away_score),
            ]:
                shot_data = self._extract_team_shots(m, side)
                synth_xg = self._compute_match_xg(
                    shot_data["shots"], shot_data["sot"],
                    shot_data["big_chances"]
                )

                self._team_xg_history[team_name].append({
                    "xg": synth_xg,
                    "goals": goals or 0,
                    "date": m.commence_time,
                })

                # Track global averages for defaults
                if side == "home":
                    all_xg_home.append(synth_xg)
                else:
                    all_xg_away.append(synth_xg)

                # Trim to window
                if len(self._team_xg_history[team_name]) > self.rolling_window * 2:
                    self._team_xg_history[team_name] = \
                        self._team_xg_history[team_name][-self.rolling_window * 2:]

        # Compute defaults from actual data
        self._global_avg_home = sum(all_xg_home) / len(all_xg_home) if all_xg_home else 0.0
        self._global_avg_away = sum(all_xg_away) / len(all_xg_away) if all_xg_away else 0.0

        self._is_fitted = True
        cal_label = "calibrated" if self.calibrated else "default-coefs"
        logger.info(
            f"SyntheticXGBuilder ({cal_label}): {len(self._team_xg_history)} teams profiled | "
            f"avg xG: home={self._global_avg_home:.3f}, away={self._global_avg_away:.3f}"
        )

    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get synthetic xG features for a match."""
        # Defaults from actual computed global averages
        avg_home = getattr(self, '_global_avg_home', 0.0)
        avg_away = getattr(self, '_global_avg_away', 0.0)
        defaults = {
            "synth_xg_home": round(avg_home, 3),
            "synth_xg_away": round(avg_away, 3),
            "synth_xg_diff": round(avg_home - avg_away, 3),
            "synth_xg_overperform_home": 0.0,
            "synth_xg_overperform_away": 0.0,
        }

        if not match:
            return defaults

        features = {}
        ref_time = as_of or match.commence_time

        for side, team_name in [
            ("home", match.home_team.normalized_name),
            ("away", match.away_team.normalized_name),
        ]:
            history = self._team_xg_history.get(team_name, [])
            if ref_time:
                history = [h for h in history if h["date"] < ref_time]
            recent = history[-self.rolling_window:]

            if recent:
                avg_xg = sum(h["xg"] for h in recent) / len(recent)
                avg_goals = sum(h["goals"] for h in recent) / len(recent)
                overperform = round(avg_goals - avg_xg, 3)
            else:
                avg_xg = defaults[f"synth_xg_{side}"]
                overperform = 0.0

            features[f"synth_xg_{side}"] = round(avg_xg, 3)
            features[f"synth_xg_overperform_{side}"] = overperform

        features["synth_xg_diff"] = round(
            features["synth_xg_home"] - features["synth_xg_away"], 3
        )

        return features
