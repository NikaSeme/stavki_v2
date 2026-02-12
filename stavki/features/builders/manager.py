"""
Manager Feature Builder (Tier 3).

Features based on manager/coach data:
- Tenure in days → longer = more stable
- New manager bounce → elevated performance in first 6-8 weeks
- Manager change signal
"""

from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict
import logging

from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

# New manager bounce: ~6-8 week effect observed in academic literature
NEW_MANAGER_BOUNCE_DAYS = 56  # 8 weeks


class ManagerFeatureBuilder:
    """
    Compute manager tenure and new-manager-bounce features.
    
    Uses enrichment.home_coach / away_coach data.
    """
    
    name = "manager"
    
    def __init__(self):
        # team -> { coach_name, appointed_date }
        self._team_coaches: Dict[str, Dict] = {}
        self._is_fitted = False
    
    def fit(self, matches: List[Match]) -> None:
        """Track coach changes per team."""
        self._team_coaches.clear()
        
        for m in sorted(matches, key=lambda x: x.commence_time):
            if not m.enrichment:
                continue
            
            if m.enrichment.home_coach:
                team = m.home_team.normalized_name
                coach = m.enrichment.home_coach
                self._team_coaches[team] = {
                    "name": coach.name,
                    "appointed": coach.appointed_date,
                    "last_seen": m.commence_time.isoformat(),
                }
            
            if m.enrichment.away_coach:
                team = m.away_team.normalized_name
                coach = m.enrichment.away_coach
                self._team_coaches[team] = {
                    "name": coach.name,
                    "appointed": coach.appointed_date,
                    "last_seen": m.commence_time.isoformat(),
                }
        
        self._is_fitted = True
        logger.info(f"ManagerFeatureBuilder: {len(self._team_coaches)} teams profiled")
    
    def _get_tenure_days(self, coach_data: dict, as_of: datetime) -> Optional[int]:
        """Calculate tenure in days from appointed date to as_of."""
        if not coach_data or not coach_data.get("appointed"):
            return None
        try:
            appointed = datetime.fromisoformat(coach_data["appointed"])
            delta = as_of - appointed
            return max(int(delta.total_seconds() / 86400), 0)
        except (ValueError, TypeError):
            return None
    
    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Get manager features for a match."""
        defaults = {
            "manager_tenure_home": 365.0,   # Assume ~1 year (neutral)
            "manager_tenure_away": 365.0,
            "new_manager_bounce_home": 0.0,
            "new_manager_bounce_away": 0.0,
        }
        
        if not match:
            return defaults
        
        ref_time = as_of or match.commence_time
        features = {}
        
        for side, team_name in [("home", match.home_team.normalized_name),
                                 ("away", match.away_team.normalized_name)]:
            coach_data = None
            
            # From enrichment
            if match.enrichment:
                coach = match.enrichment.home_coach if side == "home" else match.enrichment.away_coach
                if coach:
                    coach_data = {
                        "name": coach.name,
                        "appointed": coach.appointed_date,
                    }
            
            # From fitted history
            if not coach_data:
                coach_data = self._team_coaches.get(team_name)
            
            tenure = self._get_tenure_days(coach_data, ref_time)
            
            if tenure is not None:
                features[f"manager_tenure_{side}"] = float(tenure)
                # New manager bounce: decays linearly over 8 weeks
                if tenure <= NEW_MANAGER_BOUNCE_DAYS:
                    bounce = 1.0 - (tenure / NEW_MANAGER_BOUNCE_DAYS)
                    features[f"new_manager_bounce_{side}"] = round(bounce, 3)
                else:
                    features[f"new_manager_bounce_{side}"] = 0.0
            else:
                features[f"manager_tenure_{side}"] = defaults[f"manager_tenure_{side}"]
                features[f"new_manager_bounce_{side}"] = 0.0
        
        return features
