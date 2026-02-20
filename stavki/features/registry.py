"""
Feature Registry.

Central location for all feature builders.
Provides easy access to features by name.

Usage:
    from stavki.features.registry import FeatureRegistry
    
    registry = FeatureRegistry()
    registry.fit(historical_matches)
    
    features = registry.compute(home_team, away_team, as_of=match_date)
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import logging

from stavki.data.schemas import Match

from .builders.elo import EloBuilder
from .builders.form import FormBuilder, GoalsBuilder
from .builders.h2h import H2HBuilder
from .builders.disagreement import DisagreementBuilder
from .builders.advanced_stats import AdvancedFeatureBuilder
from .builders.roster import RosterFeatureBuilder

# Tier 1
from .builders.referee import RefereeFeatureBuilder
from .builders.player_impact import PlayerImpactFeatureBuilder
from .builders.injuries import InjuryFeatureBuilder
from .builders.injuries import InjuryFeatureBuilder
from .builders.real_xg import RealXGBuilder
# from .builders.synth_xg import SyntheticXGBuilder # Deprecated

# Tier 2
from .builders.formation import FormationFeatureBuilder
from .builders.venue import VenueFeatureBuilder
from .builders.weather import WeatherFeatureBuilder

# Tier 3 (New)
from .builders.manager import ManagerFeatureBuilder
from .builders.sm_odds import SMOddsFeatureBuilder
from .builders.seasonal import SeasonalFeatureBuilder
from .builders.corners import CornersFeatureBuilder

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Registry of all feature builders.
    
    Manages feature builders and computes features for matches.
    """
    
    def __init__(
        self,
        elo_params: Optional[Dict] = None,
        form_window: int = 5,
        goals_window: int = 10,
        h2h_max_meetings: int = 10,
        advanced_window: int = 5,
        roster_window: int = 20,
        corners_window: int = 10,
        training_mode: bool = False,
    ):
        self.training_mode = training_mode
        
        # Core builders (always active)
        self.elo = EloBuilder(**(elo_params or {}))
        self.form = FormBuilder(window=form_window)
        self.goals = GoalsBuilder(window=goals_window)
        self.h2h = H2HBuilder(max_meetings=h2h_max_meetings)
        self.advanced = AdvancedFeatureBuilder(window=advanced_window)
        self.disagreement = DisagreementBuilder()
        
        # Tier 1 — Referee + SynthXG work from CSV data; others need API
        self.referee = RefereeFeatureBuilder()
        # Tier 1 — Referee + RealXG (Hybrid Strategy)
        self.referee = RefereeFeatureBuilder()
        self.real_xg = RealXGBuilder()
        # self.synth_xg = SyntheticXGBuilder() # Deprecated
        
        # Tier 3 — CSV supported
        self.corners = CornersFeatureBuilder(rolling_window=corners_window)
        self.seasonal = SeasonalFeatureBuilder()
        
        # API-dependent builders — only initialize if not in training mode
        # (these produce constant defaults from historical CSV and add noise)
        if not training_mode:
            self.roster = RosterFeatureBuilder(window=roster_window)
            self.player_impact = PlayerImpactFeatureBuilder()
            self.injuries = InjuryFeatureBuilder()
            self.formation = FormationFeatureBuilder()
            self.venue = VenueFeatureBuilder()
            self.weather = WeatherFeatureBuilder()
            self.manager = ManagerFeatureBuilder()
            self.sm_odds = SMOddsFeatureBuilder()
        else:
            self.roster = None
            self.player_impact = None
            self.injuries = None
            self.formation = None
            self.venue = None
            self.weather = None
            self.manager = None
            self.sm_odds = None
        
        # Cache
        self._historical_matches: List[Match] = []
        self._is_fitted = False
    
    def fit(self, matches: List[Match]) -> None:
        """
        Fit builders on historical match data.
        """
        # Sort by date
        sorted_matches = sorted(
            [m for m in matches if m.is_completed],
            key=lambda m: m.commence_time
        )
        
        # Fit core builders (always)
        self.elo.fit(sorted_matches)
        self.form.fit(sorted_matches) # Form/goals now need explicit fit for indexing
        self.goals.fit(sorted_matches)
        self.advanced.fit(sorted_matches) # Pre-index advanced stats
        self.corners.fit(sorted_matches)
        
        self.referee.fit(sorted_matches)
        self.referee.fit(sorted_matches)
        self.real_xg.fit(sorted_matches)
        
        # Seasonal needs no fitting but good to init
        self.seasonal.fit(sorted_matches)
        
        # Fit API-dependent builders only if initialized
        if self.roster:
            self.roster.fit(sorted_matches)
        if self.player_impact:
            self.player_impact.fit(sorted_matches)
        if self.injuries:
            self.injuries.fit(sorted_matches)
        if self.formation:
            self.formation.fit(sorted_matches)
        if self.venue:
            self.venue.fit(sorted_matches)
        if self.manager:
            self.manager.fit(sorted_matches)
        
        # Store for other builders
        self._historical_matches = sorted_matches
        self._is_fitted = True
        
        mode_label = "training" if self.training_mode else "full"
        logger.info(f"FeatureRegistry ({mode_label}) fitted on {len(sorted_matches)} matches")
    
    def compute(
        self,
        home_team: str,
        away_team: str,
        as_of: Optional[datetime] = None,
        include_disagreement: bool = False,
        model_probs: Optional[Dict[str, List[float]]] = None,
        market_probs: Optional[List[float]] = None,
        match: Optional[Match] = None,
    ) -> Dict[str, float]:
        """
        Compute all features for a match.
        """
        if not self._is_fitted:
            raise ValueError("FeatureRegistry not fitted. Call fit() first.")
        
        features = {}
        
        # === Core Features ===
        
        # ELO features
        elo_features = self.elo.get_features(home_team, away_team, as_of)
        features.update(elo_features)
        
        # Form features
        form_features = self.form.get_features(
            match=match, as_of=as_of
        )
        features.update(form_features)
        
        # Goals features
        goals_features = self.goals.get_features(
            match=match, as_of=as_of
        )
        features.update(goals_features)
        
        # H2H features
        h2h_features = self.h2h.get_features(
            home_team, away_team, self._historical_matches, as_of
        )
        features.update(h2h_features)
        
        # Advanced Features (xG, Shots) — uses pre-indexed lookups
        # Note: compute_for_team signature was simplified to (team, matches, as_of)
        # But wait, AdvancedFeatureBuilder.compute_for_team now expects (team, matches=None, as_of=None)
        # Registry assumes old signature? No, updated code used named args.
        adv_home = self.advanced.compute_for_team(home_team, as_of=as_of)
        adv_away = self.advanced.compute_for_team(away_team, as_of=as_of)
        
        for k, v in adv_home.items():
            features[f"advanced_{k}_home"] = v
        for k, v in adv_away.items():
            features[f"advanced_{k}_away"] = v
            
        # Roster Features (requires match object with lineups)
        if self.roster and match and match.lineups:
            roster_features = self.roster.get_features(match, as_of)
            features.update(roster_features)
        
        # === Tier 1: High Impact ===
        
        # Referee features (works from CSV Referee column)
        ref_features = self.referee.get_features(match=match, as_of=as_of)
        features.update(ref_features)
        
        # Player Impact (requires lineups — API only)
        if self.player_impact and match and match.lineups:
            pi_features = self.player_impact.get_features(match=match, as_of=as_of)
            features.update(pi_features)
        
        # Injuries (API only)
        if self.injuries:
            inj_features = self.injuries.get_features(match=match, as_of=as_of)
            features.update(inj_features)
        
        # Real xG (Hybrid Strategy)
        xg_features = self.real_xg.get_features(match=match, as_of=as_of)
        features.update(xg_features)
        
        # === Tier 2: Medium Impact (API only) ===
        
        if self.formation:
            fmt_features = self.formation.get_features(match=match, as_of=as_of)
            features.update(fmt_features)
        
        if self.venue:
            venue_features = self.venue.get_features(match=match, as_of=as_of)
            features.update(venue_features)
        
        if self.weather:
            weather_features = self.weather.get_features(match=match, as_of=as_of)
            features.update(weather_features)
        
        # === Tier 3: Nice to Have ===
        
        # Seasonal (CSV supported)
        seasonal_features = self.seasonal.get_features(match=match, as_of=as_of)
        features.update(seasonal_features)
        
        # Corners (CSV supported)
        corners_features = self.corners.get_features(match=match, as_of=as_of)
        features.update(corners_features)
        
        if self.manager:
            mgr_features = self.manager.get_features(match=match, as_of=as_of)
            features.update(mgr_features)
        
        if self.sm_odds:
            sm_features = self.sm_odds.get_features(match=match, as_of=as_of)
            features.update(sm_features)
        
        # Disagreement (if model probs provided)
        if include_disagreement and model_probs:
            disagree_features = self.disagreement.get_features(
                model_probs.get("poisson", [0.33, 0.33, 0.34]),
                model_probs.get("catboost", [0.33, 0.33, 0.34]),
                model_probs.get("neural", [0.33, 0.33, 0.34]),
                market_probs
            )
            features.update(disagree_features)
        
        return features
    
    def compute_batch(
        self,
        matches: List[Match],
        as_of: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Compute features for multiple matches.
        """
        rows = []
        
        for match in matches:
            try:
                match_as_of = as_of or match.commence_time
                features = self.compute(
                    match.home_team.normalized_name,
                    match.away_team.normalized_name,
                    match_as_of,
                    match=match  # Pass match object for enriched features
                )
                features["match_id"] = match.id
                rows.append(features)
            except Exception as e:
                logger.warning(f"Failed to compute features for {match.id}: {e}")
                continue
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        return df.set_index("match_id")
    
    def get_all_feature_names(self) -> List[str]:
        """
        Get canonical list of all feature names produced by the registry.
        Acts as the Single Source of Truth for model feature lists.
        """
        # Base features that don't depend on specific builders
        # (Though ideally even these should come from builders)
        # For now, we simulate a dummy computation to get keys, 
        # but a cleaner way would be to ask builders directly.
        
        # Since builders are stateful/complex, we'll use the dummy compute method
        # but cache the result class-level if possible, or just recompute (fast enough).
        
        # We need to handle the "enriched" features manually if they require match data
        # that the dummy compute doesn't provide.
        
        dummy_names = self.get_feature_names()
        
        # Ensure consistent order
        return sorted(dummy_names)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names (internal helper)."""
        # Hack to allow getting names without fitting
        was_fitted = self._is_fitted
        self._is_fitted = True
        
        try:
            # Compute for dummy match to get keys
            # We use empty lists/defaults where possible
            sample = self.compute(
                "sample_team_a",
                "sample_team_b",
                datetime.now(),
                model_probs={"poisson": [0.33, 0.33, 0.34]}, # dummy probs for disagreement
                market_probs={"home": 0.33}
            )
            keys = list(sample.keys())
        except Exception as e:
            logger.warning(f"Failed to compute sample features during name discovery: {e}")
            # Fallback to hardcoded list if dynamic discovery fails
            keys = [
                "elo_home", "elo_away", "elo_diff",
                "form_home_pts", "form_away_pts", "form_diff",
                "form_home_gf", "form_away_gf", "gf_diff", "ga_diff",
                "advanced_xg_home", "advanced_xg_away",
                "xg_home", "xg_away", "xg_efficiency_home",
                "ref_strictness_t1", "ref_goals_zscore"
                "ref_strictness_t1", "ref_goals_zscore"
            ]
        finally:
            self._is_fitted = was_fitted

        # Manually add features that might be missing from sample (e.g. requires lineups)
        # This acts as the manual registry for now until builders expose .feature_names property
        enriched_names = [
            "roster_regularity_home", "roster_experience_home",
            "roster_regularity_away", "roster_experience_away", 
            "xi_xg90_home", "xi_xg90_away",
            "xi_strength_ratio_home", "xi_strength_ratio_away",
             # Player Impact
            "avg_rating_home", "avg_rating_away", "rating_delta",
            "key_players_home", "key_players_away",
            "xi_experience_home", "xi_experience_away",
            # Formation
            "formation_score_home", "formation_score_away",
            "formation_mismatch", "formation_is_known",
             # Venue
            "home_advantage", "is_neutral",
             # Weather
             "temp_c", "wind_speed", "humidity", "precip_mm",
             # Disagreement
             "disagreement_score", "contrarian_index"
        ]
        
        for name in enriched_names:
            if name not in keys:
                keys.append(name)
                
        return keys
    
    def get_elo_rating(self, team: str, as_of: Optional[datetime] = None) -> float:
        """Get ELO rating for a team."""
        return self.elo.get_rating(team, as_of)


# Convenience function
def build_features(
    matches: List[Match],
    home_team: str,
    away_team: str,
    as_of: Optional[datetime] = None,
    match: Optional[Match] = None
) -> Dict[str, float]:
    """
    Quick feature computation without registry setup.
    """
    registry = FeatureRegistry()
    registry.fit(matches)
    return registry.compute(home_team, away_team, as_of, match=match)
