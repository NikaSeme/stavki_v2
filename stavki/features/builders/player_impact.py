"""
Player Impact Feature Builder (Tier 1).

Tracks individual player contributions:
- Per-player rating from match details (lineups.details.type)
- Rolling player rating profiles
- Starting XI aggregate rating/strength
- Key player absence detection
- XI experience (minutes proxy for fitness)
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import logging
import numpy as np

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)


class PlayerImpactFeatureBuilder:
    """
    Compute player-level impact features.
    
    Pre-computes per-player stats from historical lineups + match results,
    then evaluates the strength of a given starting XI.
    
    Enhanced to use actual player ratings from lineups.details.type
    when available, falling back to xG share proxy for older fixtures.
    """
    
    name = "player_impact"
    
    def __init__(self, min_appearances: int = 3, rating_window: int = 10):
        self.min_appearances = min_appearances
        self.rating_window = rating_window
        
        # player_id -> { team, starts, ratings[], xg_share_total, goals_share_total, minutes }
        self._player_stats: Dict[str, Dict] = defaultdict(lambda: {
            "team": "", "starts": 0, "ratings": [],
            "xg_share_total": 0.0, "goals_share_total": 0.0, "minutes": 0,
        })
        # team -> best known XI xG/90 sum
        self._team_best_xi: Dict[str, float] = {}
        # team -> list of recent XI player IDs (for key player detection)
        self._team_recent_xi: Dict[str, List[List[str]]] = defaultdict(list)
        self._is_fitted = False
    
    def fit(self, matches: List[Match]) -> None:
        """Build player profiles from historical matches with lineups."""
        self._player_stats.clear()
        self._team_best_xi.clear()
        self._team_recent_xi.clear()
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.lineups:
                continue
            
            home_xg = (m.stats.xg_home if m.stats else None) or (m.home_score or 0)
            away_xg = (m.stats.xg_away if m.stats else None) or (m.away_score or 0)
            home_goals = m.home_score or 0
            away_goals = m.away_score or 0
            
            for side, xi, team_name, xg, goals in [
                ("home", m.lineups.home.starting_xi if m.lineups.home else [],
                 m.home_team.normalized_name, home_xg, home_goals),
                ("away", m.lineups.away.starting_xi if m.lineups.away else [],
                 m.away_team.normalized_name, away_xg, away_goals),
            ]:
                if not xi:
                    continue
                
                n_players = len(xi)
                xg_share = xg / max(n_players, 1)
                goals_share = goals / max(n_players, 1)
                player_ids = []
                
                for p in xi:
                    ps = self._player_stats[p.id]
                    ps["team"] = team_name
                    ps["starts"] += 1
                    ps["xg_share_total"] += xg_share
                    ps["goals_share_total"] += goals_share
                    ps["minutes"] += 90
                    player_ids.append(p.id)
                    
                    # Track actual rating if available
                    p_dict = p.model_dump() if hasattr(p, 'model_dump') else {}
                    rating = p_dict.get("rating")
                    if rating is not None:
                        try:
                            ps["ratings"].append(float(rating))
                            # Keep only last N ratings
                            if len(ps["ratings"]) > self.rating_window:
                                ps["ratings"] = ps["ratings"][-self.rating_window:]
                        except (ValueError, TypeError):
                            pass
                
                # Track recent XIs for key player detection
                self._team_recent_xi[team_name].append(player_ids)
                if len(self._team_recent_xi[team_name]) > 5:
                    self._team_recent_xi[team_name] = \
                        self._team_recent_xi[team_name][-5:]
        
        # Calculate best XI per team
        team_players: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for pid, stats in self._player_stats.items():
            if stats["starts"] >= self.min_appearances:
                xg_per_90 = stats["xg_share_total"] / max(stats["starts"], 1)
                team_players[stats["team"]].append((pid, xg_per_90))
        
        for team, players in team_players.items():
            players.sort(key=lambda x: x[1], reverse=True)
            best_11 = players[:11]
            self._team_best_xi[team] = sum(xg for _, xg in best_11)
        
        self._is_fitted = True
        rated_players = sum(1 for s in self._player_stats.values() if s["ratings"])
        logger.info(f"PlayerImpactFeatureBuilder: {len(self._player_stats)} players "
                     f"({rated_players} with ratings), "
                     f"{len(self._team_best_xi)} teams profiled")
    
    def _get_xi_strength(self, team_name: str,
                         player_ids: List[str]) -> Tuple[float, float]:
        """Calculate aggregate xG/90 and strength ratio for a starting XI."""
        xi_xg = 0.0
        known_count = 0
        
        for pid in player_ids:
            if pid in self._player_stats and \
               self._player_stats[pid]["starts"] >= self.min_appearances:
                stats = self._player_stats[pid]
                xi_xg += stats["xg_share_total"] / max(stats["starts"], 1)
                known_count += 1
        
        # Fill unknown players with team average
        if known_count < len(player_ids) and team_name in self._team_best_xi:
            avg_per_player = self._team_best_xi[team_name] / 11
            xi_xg += avg_per_player * (len(player_ids) - known_count)
        
        # Strength ratio
        best = self._team_best_xi.get(team_name, xi_xg)
        ratio = xi_xg / max(best, 0.01)
        
        return xi_xg, min(ratio, 1.5)
    
    def _get_xi_ratings(self, player_ids: List[str]) -> Tuple[float, float, int]:
        """
        Get average and variance of player ratings for a starting XI.
        
        Returns: (avg_rating, rating_variance, key_player_count)
        """
        ratings = []
        key_count = 0
        
        for pid in player_ids:
            ps = self._player_stats.get(pid)
            if ps and ps["ratings"]:
                avg_r = np.mean(ps["ratings"][-self.rating_window:])
                ratings.append(avg_r)
                if avg_r >= 7.5:
                    key_count += 1
        
        if ratings:
            return float(np.mean(ratings)), float(np.std(ratings)), key_count
        return 6.5, 0.0, 0  # defaults
    
    def _detect_key_player_missing(self, team_name: str,
                                    current_ids: List[str]) -> int:
        """
        Detect if a key player (rated >= 7.5) who regularly starts
        is missing from the current XI.
        
        Returns count of missing key players.
        """
        recent_xis = self._team_recent_xi.get(team_name, [])
        if len(recent_xis) < 3:
            return 0
        
        # Find regulars: players who started in >= 60% of recent matches
        from collections import Counter
        all_ids = [pid for xi in recent_xis[-5:] for pid in xi]
        counts = Counter(all_ids)
        threshold = len(recent_xis[-5:]) * 0.6
        
        regulars = {pid for pid, c in counts.items() if c >= threshold}
        current_set = set(current_ids)
        
        missing_key = 0
        for pid in regulars - current_set:
            ps = self._player_stats.get(pid)
            if ps and ps["ratings"]:
                avg_r = np.mean(ps["ratings"][-self.rating_window:])
                if avg_r >= 7.5:
                    missing_key += 1
        
        return missing_key
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get player impact features for a match."""
        defaults = {
            "xi_xg90_home": 0.0,
            "xi_xg90_away": 0.0,
            "xi_strength_ratio_home": 1.0,
            "xi_strength_ratio_away": 1.0,
            "avg_rating_xi_home": 6.5,
            "avg_rating_xi_away": 6.5,
            "rating_delta": 0.0,
            "key_player_missing_home": 0,
            "key_player_missing_away": 0,
            "xi_rating_variance_home": 0.0,
            "xi_rating_variance_away": 0.0,
        }
        
        if not match or not match.lineups:
            return defaults
        
        features = {}
        
        for side, lineup, team_name in [
            ("home", match.lineups.home, match.home_team.normalized_name),
            ("away", match.lineups.away, match.away_team.normalized_name),
        ]:
            ids = [p.id for p in lineup.starting_xi] if lineup and lineup.starting_xi else []
            
            if ids:
                # xG-based strength (original features)
                xg_val, ratio = self._get_xi_strength(team_name, ids)
                features[f"xi_xg90_{side}"] = round(xg_val, 3)
                features[f"xi_strength_ratio_{side}"] = round(ratio, 3)
                
                # Rating-based features (new)
                avg_r, var_r, key_count = self._get_xi_ratings(ids)
                features[f"avg_rating_xi_{side}"] = round(avg_r, 2)
                features[f"xi_rating_variance_{side}"] = round(var_r, 3)
                
                # Key player absence
                missing = self._detect_key_player_missing(team_name, ids)
                features[f"key_player_missing_{side}"] = missing
            else:
                for k in ["xi_xg90", "xi_strength_ratio", "avg_rating_xi",
                           "xi_rating_variance", "key_player_missing"]:
                    features[f"{k}_{side}"] = defaults[f"{k}_{side}"]
        
        # Rating differential
        features["rating_delta"] = round(
            features.get("avg_rating_xi_home", 6.5) -
            features.get("avg_rating_xi_away", 6.5), 2
        )
        
        return features
