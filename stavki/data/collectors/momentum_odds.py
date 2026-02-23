"""
Momentum Odds Collector.

Captures a multi-day time-series of Odds from The Odds API for upcoming matches.
Unlike ClosingOddsCollector (which runs only for matches <15m from kickoff),
this daemon targets matches occurring 1-48 hours in the future.

Designed to be scheduled every 4 hours to build a trajectory of market push/pull 
(Line Momentum) leading up to kickoff.
"""

import logging
from datetime import datetime, timezone
from typing import List

from .odds_api import OddsAPIClient
from ..schemas import League
from ..storage.database import Database

logger = logging.getLogger(__name__)

# How far ahead to start tracking odds
MAX_HOURS_AHEAD = 48
# Only track matches that are at least this far away
# (Matches closer than 1 hr are the domain of ClosingOddsCollector / LivePredictor)
MIN_HOURS_AHEAD = 1


class MomentumOddsCollector:
    """
    Captures periodic line momentum updates for upcoming matches.

    Usage:
        collector = MomentumOddsCollector()
        captured = collector.capture()
        # Returns number of matches whose odds were updated
    """

    def __init__(
        self,
        db: Database = None,
        odds_client: OddsAPIClient = None,
        min_hours: int = MIN_HOURS_AHEAD,
        max_hours: int = MAX_HOURS_AHEAD,
    ):
        self.db = db or Database()
        self.odds_client = odds_client or OddsAPIClient()
        self.min_hours = min_hours
        self.max_hours = max_hours

    def capture(self) -> int:
        """
        Main entry point. Capture momentum odds for all active leagues.

        Returns:
            Number of matches updated.
        """
        now = datetime.now(timezone.utc)
        captured_count = 0

        for league, sport_key in OddsAPIClient.SPORT_KEYS.items():
            if league == League.NBA:
                continue  # Football only for now

            try:
                count = self._capture_league(league, sport_key, now)
                captured_count += count
            except Exception as e:
                logger.error(f"Failed to capture momentum odds for {league.value}: {e}")

        if captured_count > 0:
            logger.info(f"Captured momentum odds for {captured_count} upcoming matches")

        return captured_count

    def _capture_league(
        self,
        league: League,
        sport_key: str,
        now: datetime,
    ) -> int:
        """Fetch and inject odds into the database for a single league."""
        
        # Fresh API request, bypass cache to get the newest ticks
        response = self.odds_client.get_odds(sport_key)
        if not response.success:
            logger.warning(f"API error for momentum scan {league.value}: {response.error}")
            return 0
            
        # Clear local cache immediately
        self.odds_client._cache.clear()

        updated_matches = 0

        for event in response.data or []:
            commence_time = datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            )

            time_to_kickoff_hours = (commence_time - now).total_seconds() / 3600
            
            # Filter matches strictly between our bounds (e.g., 1hr to 48hr from now)
            if time_to_kickoff_hours < self.min_hours or time_to_kickoff_hours > self.max_hours:
                continue

            # Parse the event's odds
            from .closing_odds import ClosingOddsCollector
            snapshots = ClosingOddsCollector._parse_event_odds(event)
            
            if not snapshots:
                continue
                
            # Persist each bookmaker snapshot directly to the SQLite DB
            for snapshot in snapshots:
                self.db.save_odds_snapshot(snapshot)
                
            updated_matches += 1
            logger.debug(f"Saved {len(snapshots)} momentum snapshots for {event.get('home_team')} vs {event.get('away_team')} ({time_to_kickoff_hours:.1f}h to KO)")

        return updated_matches
