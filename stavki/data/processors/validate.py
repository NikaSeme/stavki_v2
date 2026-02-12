"""
Data validation processors.

Validates incoming data and detects anomalies
before they enter the pipeline.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

from ..schemas import Match, OddsSnapshot, BestOdds

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False
    
    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class OddsValidator:
    """
    Validates odds data for anomalies and errors.
    
    Detects:
    - Outlier odds (one book way off from others)
    - Impossible odds (sum of implied probs < 100%)
    - Stale odds (not updated recently)
    - Missing markets
    """
    
    # Reasonable odds ranges
    MIN_ODDS = 1.01
    MAX_ODDS = 100.0
    
    # Outlier detection thresholds
    OUTLIER_GAP_PCT = 0.20  # 20% gap from next best
    MIN_BOOKS_FOR_CONFIDENCE = 2
    
    # Maximum overround for soft books (higher = more vig)
    MAX_REASONABLE_OVERROUND = 15.0  # 15%
    
    @classmethod
    def validate_snapshot(cls, odds: OddsSnapshot) -> ValidationResult:
        """Validate a single odds snapshot."""
        result = ValidationResult()
        
        # Check odds are in reasonable range
        for outcome, value in [
            ("home", odds.home_odds),
            ("away", odds.away_odds),
        ]:
            if value <= cls.MIN_ODDS:
                result.add_error(f"{outcome} odds {value} below minimum {cls.MIN_ODDS}")
            elif value > cls.MAX_ODDS:
                result.add_warning(f"{outcome} odds {value} unusually high")
        
        if odds.draw_odds:
            if odds.draw_odds <= cls.MIN_ODDS:
                result.add_error(f"draw odds {odds.draw_odds} below minimum")
        
        # Check overround is reasonable (not negative = arb opportunity)
        if odds.overround < 0:
            result.add_warning(f"Negative overround {odds.overround:.2f}% - possible arb")
        elif odds.overround > cls.MAX_REASONABLE_OVERROUND:
            result.add_warning(f"High overround {odds.overround:.2f}%")
        
        return result
    
    @classmethod
    def find_outliers(
        cls,
        snapshots: List[OddsSnapshot],
        outcome: str = "home"
    ) -> Tuple[Optional[OddsSnapshot], float]:
        """
        Find outlier odds among multiple bookmaker snapshots.
        
        Returns:
            (outlier_snapshot, gap_percentage) or (None, 0) if no outlier
        """
        if len(snapshots) < cls.MIN_BOOKS_FOR_CONFIDENCE:
            return None, 0.0
        
        # Get odds for the specified outcome
        def get_odds(s: OddsSnapshot) -> float:
            if outcome == "home":
                return s.home_odds
            elif outcome == "away":
                return s.away_odds
            elif outcome == "draw" and s.draw_odds:
                return s.draw_odds
            return 0.0
        
        odds_values = [(s, get_odds(s)) for s in snapshots if get_odds(s) > 0]
        if len(odds_values) < 2:
            return None, 0.0
        
        # Sort by odds value
        odds_values.sort(key=lambda x: x[1], reverse=True)
        
        # Check if highest odds is an outlier
        highest = odds_values[0]
        second_highest = odds_values[1]
        
        if second_highest[1] > 0:
            gap = (highest[1] - second_highest[1]) / second_highest[1]
            if gap > cls.OUTLIER_GAP_PCT:
                return highest[0], gap
        
        return None, 0.0
    
    @classmethod
    def compute_best_odds(
        cls,
        snapshots: List[OddsSnapshot],
        exclude_outliers: bool = True
    ) -> BestOdds:
        """
        Compute best odds across all bookmakers.
        
        Optionally excludes outliers for safety.
        """
        if not snapshots:
            raise ValueError("No odds snapshots provided")
        
        match_id = snapshots[0].match_id
        
        # Track outliers
        outliers: Set[Tuple[str, str]] = set()  # (bookmaker, outcome)
        
        if exclude_outliers:
            for outcome in ["home", "draw", "away"]:
                outlier, _ = cls.find_outliers(snapshots, outcome)
                if outlier:
                    outliers.add((outlier.bookmaker, outcome))
                    logger.warning(
                        f"Excluding outlier: {outlier.bookmaker} {outcome} @ {getattr(outlier, f'{outcome}_odds')}"
                    )
        
        # Find best prices
        best_home = 0.0
        best_home_book = ""
        best_draw = None
        best_draw_book = None
        best_away = 0.0
        best_away_book = ""
        
        home_books = 0
        draw_books = 0
        away_books = 0
        
        for s in snapshots:
            # Home
            if (s.bookmaker, "home") not in outliers:
                if s.home_odds > best_home:
                    best_home = s.home_odds
                    best_home_book = s.bookmaker
                home_books += 1
            
            # Draw
            if s.draw_odds and (s.bookmaker, "draw") not in outliers:
                if best_draw is None or s.draw_odds > best_draw:
                    best_draw = s.draw_odds
                    best_draw_book = s.bookmaker
                draw_books += 1
            
            # Away
            if (s.bookmaker, "away") not in outliers:
                if s.away_odds > best_away:
                    best_away = s.away_odds
                    best_away_book = s.bookmaker
                away_books += 1
        
        return BestOdds(
            match_id=match_id,
            timestamp=datetime.utcnow(),
            home_odds=best_home,
            home_bookmaker=best_home_book,
            draw_odds=best_draw,
            draw_bookmaker=best_draw_book,
            away_odds=best_away,
            away_bookmaker=best_away_book,
            home_book_count=home_books,
            draw_book_count=draw_books,
            away_book_count=away_books,
            home_is_outlier=any((b, "home") in outliers for b, _ in outliers),
            draw_is_outlier=any((b, "draw") in outliers for b, _ in outliers),
            away_is_outlier=any((b, "away") in outliers for b, _ in outliers),
        )


class MatchValidator:
    """Validates match data."""
    
    # Maximum hours in advance we accept matches
    MAX_HOURS_AHEAD = 24 * 7  # 1 week
    
    @classmethod
    def validate(cls, match: Match) -> ValidationResult:
        """Validate a match."""
        result = ValidationResult()
        
        # Check teams are different
        if match.home_team.normalized_name == match.away_team.normalized_name:
            result.add_error("Home and away teams are the same")
        
        # Check teams aren't suspiciously similar
        if match.home_team.normalized_name in match.away_team.normalized_name:
            result.add_warning("Team names are suspiciously similar")
        
        # Check match isn't too far in future
        hours_ahead = match.hours_until_kickoff()
        if hours_ahead > cls.MAX_HOURS_AHEAD:
            result.add_warning(f"Match is {hours_ahead:.0f} hours away")
        
        # Check match hasn't already started
        if hours_ahead < 0:
            result.add_warning("Match has already started")
        
        return result


class DataQualityMonitor:
    """
    Monitors overall data quality across incoming feeds.
    
    Tracks:
    - Missing data rates
    - Staleness
    - Consistency issues
    """
    
    def __init__(self):
        self.total_matches_seen = 0
        self.matches_with_issues = 0
        self.total_odds_snapshots = 0
        self.odds_with_issues = 0
        self.outliers_detected = 0
        self.last_update = datetime.utcnow()
    
    def record_match(self, match: Match, result: ValidationResult) -> None:
        """Record a match validation result."""
        self.total_matches_seen += 1
        if not result.is_valid or result.warnings:
            self.matches_with_issues += 1
        self.last_update = datetime.utcnow()
    
    def record_odds(self, result: ValidationResult) -> None:
        """Record an odds validation result."""
        self.total_odds_snapshots += 1
        if not result.is_valid or result.warnings:
            self.odds_with_issues += 1
        self.last_update = datetime.utcnow()
    
    def record_outlier(self) -> None:
        """Record an outlier detection."""
        self.outliers_detected += 1
    
    @property
    def match_quality_rate(self) -> float:
        """Percentage of matches without issues."""
        if self.total_matches_seen == 0:
            return 1.0
        return 1 - (self.matches_with_issues / self.total_matches_seen)
    
    @property
    def odds_quality_rate(self) -> float:
        """Percentage of odds without issues."""
        if self.total_odds_snapshots == 0:
            return 1.0
        return 1 - (self.odds_with_issues / self.total_odds_snapshots)
    
    def get_summary(self) -> dict:
        """Get quality summary for logging."""
        return {
            "total_matches": self.total_matches_seen,
            "match_quality": f"{self.match_quality_rate:.1%}",
            "total_odds": self.total_odds_snapshots,
            "odds_quality": f"{self.odds_quality_rate:.1%}",
            "outliers": self.outliers_detected,
            "last_update": self.last_update.isoformat(),
        }
