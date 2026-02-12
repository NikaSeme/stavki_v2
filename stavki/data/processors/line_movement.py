"""
Line movement tracking and analysis.

Tracks how odds change over time to detect:
- Sharp money (sudden significant drops)
- Steam moves (coordinated drops across books)
- Public bias (slow drifts)
- CLV opportunities
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ..schemas import OddsSnapshot, LineMovement, Outcome

logger = logging.getLogger(__name__)


@dataclass
class LineMovementTracker:
    """
    Tracks odds movement over time for a match.
    
    Usage:
        tracker = LineMovementTracker(match_id)
        tracker.add_snapshot(snapshot1)
        tracker.add_snapshot(snapshot2)
        
        movements = tracker.get_movements()
        # movements["home"].is_steaming -> sharp money indicator
    """
    
    match_id: str
    snapshots: List[OddsSnapshot] = field(default_factory=list)
    
    def add_snapshot(self, snapshot: OddsSnapshot) -> None:
        """Add an odds snapshot."""
        if snapshot.match_id != self.match_id:
            raise ValueError(f"Snapshot match_id mismatch: {snapshot.match_id} != {self.match_id}")
        self.snapshots.append(snapshot)
        # Keep sorted by timestamp
        self.snapshots.sort(key=lambda s: s.timestamp)
    
    def add_snapshots(self, snapshots: List[OddsSnapshot]) -> None:
        """Add multiple snapshots."""
        for s in snapshots:
            self.add_snapshot(s)
    
    def get_best_odds_over_time(self, outcome: str) -> List[Tuple[datetime, float]]:
        """
        Get best odds for an outcome at each point in time.
        
        Returns list of (timestamp, best_odds) tuples.
        """
        result = []
        
        # Group snapshots by timestamp (within 1 minute)
        time_groups: Dict[str, List[OddsSnapshot]] = {}
        for s in self.snapshots:
            key = s.timestamp.strftime("%Y%m%d%H%M")
            if key not in time_groups:
                time_groups[key] = []
            time_groups[key].append(s)
        
        # Find best odds in each group
        for key, group in sorted(time_groups.items()):
            best_odds = 0.0
            for s in group:
                odds = getattr(s, f"{outcome}_odds", None)
                if odds and odds > best_odds:
                    best_odds = odds
            
            if best_odds > 0:
                result.append((group[0].timestamp, best_odds))
        
        return result
    
    def get_movement(self, outcome: str) -> Optional[LineMovement]:
        """
        Get line movement for an outcome.
        """
        history = self.get_best_odds_over_time(outcome)
        
        if len(history) < 2:
            return None
        
        opening = history[0]
        current = history[-1]
        max_odds = max(h[1] for h in history)
        min_odds = min(h[1] for h in history)
        
        return LineMovement(
            match_id=self.match_id,
            outcome=Outcome(outcome),
            opening_odds=opening[1],
            current_odds=current[1],
            first_seen=opening[0],
            last_updated=current[0],
            num_snapshots=len(history),
            max_odds=max_odds,
            min_odds=min_odds,
        )
    
    def get_all_movements(self) -> Dict[str, LineMovement]:
        """Get line movements for all outcomes."""
        movements = {}
        for outcome in ["home", "draw", "away"]:
            mov = self.get_movement(outcome)
            if mov:
                movements[outcome] = mov
        return movements
    
    def detect_steam(self, threshold_pct: float = 10.0) -> List[str]:
        """
        Detect steam moves (sharp money indicators).
        
        Returns list of outcomes with significant drops.
        """
        steaming = []
        movements = self.get_all_movements()
        
        for outcome, mov in movements.items():
            if mov.total_movement_pct < -threshold_pct:
                steaming.append(outcome)
                logger.info(
                    f"Steam detected on {outcome}: {mov.opening_odds:.2f} -> {mov.current_odds:.2f} "
                    f"({mov.total_movement_pct:.1f}%)"
                )
        
        return steaming
    
    def detect_drift(self, threshold_pct: float = 10.0) -> List[str]:
        """
        Detect drifts (public fade / liability management).
        
        Returns list of outcomes with significant rises.
        """
        drifting = []
        movements = self.get_all_movements()
        
        for outcome, mov in movements.items():
            if mov.total_movement_pct > threshold_pct:
                drifting.append(outcome)
        
        return drifting


class SharpMoneyDetector:
    """
    Detects sharp money signals from odds movement patterns.
    
    Sharp money indicators:
    1. Sudden significant drops (5%+ in < 1 hour)
    2. Coordinated moves across multiple bookmakers
    3. Pinnacle/Betfair leading the move
    """
    
    SHARP_BOOKS = ["pinnacle", "betfair_ex_eu", "sbo", "ps3838"]
    
    @classmethod
    def is_sharp_move(
        cls,
        old_snapshot: OddsSnapshot,
        new_snapshot: OddsSnapshot,
        outcome: str,
        threshold_pct: float = 5.0,
        max_minutes: int = 60
    ) -> bool:
        """
        Check if odds movement constitutes a sharp move.
        """
        # Check timing
        time_diff = (new_snapshot.timestamp - old_snapshot.timestamp).total_seconds() / 60
        if time_diff > max_minutes:
            return False
        
        old_odds = getattr(old_snapshot, f"{outcome}_odds", None)
        new_odds = getattr(new_snapshot, f"{outcome}_odds", None)
        
        if not old_odds or not new_odds:
            return False
        
        # Calculate movement
        move_pct = ((new_odds - old_odds) / old_odds) * 100
        
        # Sharp money = odds dropping significantly and quickly
        return move_pct < -threshold_pct
    
    @classmethod
    def detect_coordinated_move(
        cls,
        snapshots_before: List[OddsSnapshot],
        snapshots_after: List[OddsSnapshot],
        outcome: str,
        min_books: int = 3,
        threshold_pct: float = 3.0
    ) -> bool:
        """
        Detect if multiple bookmakers moved in same direction.
        """
        moves_down = 0
        
        for before in snapshots_before:
            for after in snapshots_after:
                if before.bookmaker == after.bookmaker:
                    old_odds = getattr(before, f"{outcome}_odds", None)
                    new_odds = getattr(after, f"{outcome}_odds", None)
                    
                    if old_odds and new_odds:
                        move_pct = ((new_odds - old_odds) / old_odds) * 100
                        if move_pct < -threshold_pct:
                            moves_down += 1
        
        return moves_down >= min_books


class CLVTracker:
    """
    Closing Line Value tracker.
    
    CLV measures if you consistently beat the closing line.
    Positive CLV over many bets = real edge.
    """
    
    @staticmethod
    def calculate_clv(
        bet_odds: float,
        closing_odds: float
    ) -> float:
        """
        Calculate CLV percentage.
        
        Positive = you got better odds than closing
        Negative = you got worse odds than closing
        
        Returns:
            CLV as percentage (e.g., 5.0 = 5% better than closing)
        """
        return (bet_odds / closing_odds - 1) * 100
    
    @staticmethod
    def expected_roi_from_clv(clv_pct: float) -> float:
        """
        Estimate expected ROI from positive CLV.
        
        Rule of thumb: 1% CLV ≈ 1% ROI long-term
        (slightly less in practice due to vig)
        """
        return clv_pct * 0.8  # Conservative estimate
    
    @staticmethod
    def analyze_clv_distribution(clvs: List[float]) -> Dict[str, float]:
        """
        Analyze CLV distribution across bets.
        """
        if not clvs:
            return {}
        
        import statistics
        
        positive = [c for c in clvs if c > 0]
        negative = [c for c in clvs if c < 0]
        
        return {
            "mean": statistics.mean(clvs),
            "median": statistics.median(clvs),
            "std": statistics.stdev(clvs) if len(clvs) > 1 else 0,
            "positive_rate": len(positive) / len(clvs),
            "avg_positive": statistics.mean(positive) if positive else 0,
            "avg_negative": statistics.mean(negative) if negative else 0,
        }
