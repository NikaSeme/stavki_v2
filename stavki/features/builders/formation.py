"""
Formation / Tactical Feature Builder (Tier 2).

Extracts formation archetypes from lineup data:
- Defensive (5-x-x), Balanced (4-x-x), Attacking (3-x-x)
- Formation matchup matrix (historical win rates per pair)
- Tactical style indicators
- Formation diversity (tactical flexibility)
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter, defaultdict
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# Formation archetype classification
FORMATION_ARCHETYPES = {
    "defensive": {"5-4-1", "5-3-2", "5-2-3", "5-2-1-2", "4-5-1"},
    "balanced": {"4-4-2", "4-3-3", "4-2-3-1", "4-1-4-1", "4-4-1-1", "4-3-2-1"},
    "attacking": {"3-5-2", "3-4-3", "3-4-2-1", "3-3-4", "3-4-1-2"},
}


def _classify_formation(formation: Optional[str]) -> str:
    """Classify a formation string into an archetype."""
    if not formation:
        return "unknown"
    f = formation.strip()
    for archetype, patterns in FORMATION_ARCHETYPES.items():
        if f in patterns:
            return archetype
    # Fallback: check first digit
    try:
        defenders = int(f.split("-")[0])
        if defenders >= 5:
            return "defensive"
        elif defenders <= 3:
            return "attacking"
        return "balanced"
    except (ValueError, IndexError):
        return "unknown"


def _formation_score(formation: Optional[str]) -> float:
    """Score a formation on a defensive (0.0) to attacking (1.0) scale."""
    archetype = _classify_formation(formation)
    mapping = {"defensive": 0.2, "balanced": 0.5, "attacking": 0.8, "unknown": 0.5}
    return mapping[archetype]


class FormationFeatureBuilder:
    """
    Compute formation/tactical matchup features.

    Uses lineup.formation from the enrichment data.
    Builds a matchup matrix tracking historical outcomes per formation pair.
    """

    name = "formation"

    def __init__(self):
        # Track team formation preferences
        self._team_formations: Dict[str, List[str]] = {}
        # Matchup matrix: (home_formation, away_formation) -> {wins, draws, losses}
        self._matchup_matrix: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
            lambda: {"wins": 0, "draws": 0, "losses": 0}
        )
        # Formation diversity per team
        self._team_unique_formations: Dict[str, set] = defaultdict(set)
        self._is_fitted = False

    def fit(self, matches: List[Match]) -> None:
        """Track formation preferences and build matchup matrix per team."""
        self._team_formations.clear()
        self._matchup_matrix.clear()
        self._team_unique_formations.clear()

        team_fmts: Dict[str, List[str]] = {}

        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.lineups:
                continue

            home = m.home_team.normalized_name
            away = m.away_team.normalized_name

            home_fmt = m.lineups.home.formation if m.lineups.home else None
            away_fmt = m.lineups.away.formation if m.lineups.away else None

            if home_fmt:
                if home not in team_fmts:
                    team_fmts[home] = []
                team_fmts[home].append(home_fmt)
                team_fmts[home] = team_fmts[home][-10:]
                self._team_unique_formations[home].add(home_fmt)

            if away_fmt:
                if away not in team_fmts:
                    team_fmts[away] = []
                team_fmts[away].append(away_fmt)
                team_fmts[away] = team_fmts[away][-10:]
                self._team_unique_formations[away].add(away_fmt)

            # Build matchup matrix from completed matches with both formations
            if home_fmt and away_fmt and m.is_completed:
                key = (home_fmt.strip(), away_fmt.strip())
                home_score = m.home_score or 0
                away_score = m.away_score or 0

                if home_score > away_score:
                    self._matchup_matrix[key]["wins"] += 1
                elif home_score == away_score:
                    self._matchup_matrix[key]["draws"] += 1
                else:
                    self._matchup_matrix[key]["losses"] += 1

        self._team_formations = team_fmts
        self._is_fitted = True

        n_pairs = len(self._matchup_matrix)
        n_teams = len(team_fmts)
        logger.info(
            f"FormationFeatureBuilder: {n_teams} teams profiled, "
            f"{n_pairs} formation matchup pairs tracked"
        )

    def _get_preferred_style(self, team: str) -> float:
        """Get team's average formation score (defensive to attacking)."""
        fmts = self._team_formations.get(team, [])
        if not fmts:
            return 0.5
        scores = [_formation_score(f) for f in fmts]
        return sum(scores) / len(scores)

    def _get_matchup_winrate(
        self, home_fmt: Optional[str], away_fmt: Optional[str]
    ) -> float:
        """
        Get historical home win rate for a (home_formation, away_formation) pair.

        Returns 0.5 (neutral) if insufficient data.
        """
        if not home_fmt or not away_fmt:
            return 0.5

        key = (home_fmt.strip(), away_fmt.strip())
        record = self._matchup_matrix.get(key)

        if not record:
            return 0.5

        total = record["wins"] + record["draws"] + record["losses"]
        if total < 3:
            # Insufficient data — blend with 0.5 prior
            raw = (record["wins"] + 0.5 * record["draws"]) / total
            # Bayesian smoothing: blend with prior (weight by sample size)
            prior_weight = 3.0
            return (raw * total + 0.5 * prior_weight) / (total + prior_weight)

        return (record["wins"] + 0.5 * record["draws"]) / total

    def _get_formation_diversity(self, team: str) -> float:
        """
        Get formation diversity for a team (0.0-1.0 scale).

        Higher = more tactical flexibility (uses many formations).
        Log-scaled to handle varying sample sizes.
        """
        import math
        n_unique = len(self._team_unique_formations.get(team, set()))
        # Cap at 7 unique formations for normalization
        return round(min(math.log1p(n_unique) / math.log1p(7), 1.0), 3)

    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get formation features for a match."""
        defaults = {
            "formation_score_home": 0.5,
            "formation_score_away": 0.5,
            "formation_mismatch": 0.0,
            "home_style_attacking": 0.5,
            "away_style_attacking": 0.5,
            "formation_matchup_winrate": 0.5,
            "formation_diversity_home": 0.0,
            "formation_diversity_away": 0.0,
        }

        if not match:
            return defaults

        features = {}

        # Current match formations (if available from lineup)
        home_fmt = None
        away_fmt = None
        if match.lineups:
            home_fmt = match.lineups.home.formation
            away_fmt = match.lineups.away.formation

        home_score = _formation_score(home_fmt) if home_fmt else self._get_preferred_style(
            match.home_team.normalized_name
        )
        away_score = _formation_score(away_fmt) if away_fmt else self._get_preferred_style(
            match.away_team.normalized_name
        )

        features["formation_score_home"] = round(home_score, 2)
        features["formation_score_away"] = round(away_score, 2)
        features["formation_mismatch"] = round(abs(home_score - away_score), 2)
        features["home_style_attacking"] = round(
            self._get_preferred_style(match.home_team.normalized_name), 2
        )
        features["away_style_attacking"] = round(
            self._get_preferred_style(match.away_team.normalized_name), 2
        )

        # Formation matchup win rate from historical matrix
        features["formation_matchup_winrate"] = round(
            self._get_matchup_winrate(home_fmt, away_fmt), 3
        )

        # Formation diversity (tactical flexibility)
        features["formation_diversity_home"] = self._get_formation_diversity(
            match.home_team.normalized_name
        )
        features["formation_diversity_away"] = self._get_formation_diversity(
            match.away_team.normalized_name
        )

        return features
