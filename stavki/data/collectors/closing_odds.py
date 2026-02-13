"""
Closing Odds Collector for CLV (Closing Line Value) Tracking.

Captures odds from The Odds API just before match kickoff to compute CLV,
the #1 metric for assessing long-term betting profitability.

CLV = (odds_at_placement / closing_odds - 1) × 100

A positive CLV means you consistently got better odds than the market closing
price — the strongest indicator of a profitable bettor.

Strategy:
    1. Find matches kicking off within the next 15 minutes
    2. Fetch current odds from The Odds API for those matches
    3. Extract Pinnacle odds (sharpest book = gold standard for CLV)
    4. Also compute average and best odds across all bookmakers
    5. Save to the closing_odds DB table
    6. Update any pending bets with their closing odds for CLV calculation
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any

from .odds_api import OddsAPIClient, OddsAPICollector
from ..schemas import League, OddsSnapshot
from ..storage.database import Database

logger = logging.getLogger(__name__)

# Pinnacle is the sharpest bookmaker — their closing line is the benchmark
PINNACLE_KEY = "pinnacle"

# How far ahead (minutes) to capture closing odds before kickoff
CAPTURE_WINDOW_MINUTES = 15


class ClosingOddsData:
    """Closing odds data for a single match."""

    def __init__(
        self,
        match_id: str,
        captured_at: datetime,
        minutes_before_kickoff: int,
        snapshots: List[OddsSnapshot],
    ):
        self.match_id = match_id
        self.captured_at = captured_at
        self.minutes_before_kickoff = minutes_before_kickoff
        self.snapshots = snapshots

        # Extract Pinnacle odds (the CLV benchmark)
        self.pinnacle_home: Optional[float] = None
        self.pinnacle_draw: Optional[float] = None
        self.pinnacle_away: Optional[float] = None

        # Compute average and best odds across all bookmakers
        self.avg_home: float = 0.0
        self.avg_draw: Optional[float] = None
        self.avg_away: float = 0.0
        self.best_home: float = 0.0
        self.best_draw: Optional[float] = None
        self.best_away: float = 0.0

        self._compute(snapshots)

    def _compute(self, snapshots: List[OddsSnapshot]) -> None:
        """Compute Pinnacle, average, and best odds from snapshots."""
        if not snapshots:
            return

        home_odds_list = []
        draw_odds_list = []
        away_odds_list = []

        for snap in snapshots:
            home_odds_list.append(snap.home_odds)
            away_odds_list.append(snap.away_odds)
            if snap.draw_odds:
                draw_odds_list.append(snap.draw_odds)

            # Extract Pinnacle specifically
            if snap.bookmaker == PINNACLE_KEY:
                self.pinnacle_home = snap.home_odds
                self.pinnacle_draw = snap.draw_odds
                self.pinnacle_away = snap.away_odds

        if home_odds_list:
            self.avg_home = round(sum(home_odds_list) / len(home_odds_list), 3)
            self.best_home = round(max(home_odds_list), 3)

        if draw_odds_list:
            self.avg_draw = round(sum(draw_odds_list) / len(draw_odds_list), 3)
            self.best_draw = round(max(draw_odds_list), 3)

        if away_odds_list:
            self.avg_away = round(sum(away_odds_list) / len(away_odds_list), 3)
            self.best_away = round(max(away_odds_list), 3)

    def get_closing_odds_for_outcome(self, outcome: str) -> Optional[float]:
        """
        Get the best closing odds reference for a specific bet outcome.

        Priority: Pinnacle > Average (Pinnacle is sharpest).

        Args:
            outcome: "home", "draw", or "away"

        Returns:
            Closing odds value, or None if not available
        """
        outcome = outcome.lower()

        # Try Pinnacle first (gold standard)
        pinnacle_map = {
            "home": self.pinnacle_home,
            "draw": self.pinnacle_draw,
            "away": self.pinnacle_away,
        }
        if pinnacle_map.get(outcome):
            return pinnacle_map[outcome]

        # Fall back to average
        avg_map = {
            "home": self.avg_home,
            "draw": self.avg_draw,
            "away": self.avg_away,
        }
        return avg_map.get(outcome)


class ClosingOddsCollector:
    """
    Captures closing odds for matches about to kick off.

    Designed to run every 5 minutes via scheduler. On each run:
    1. Finds matches starting within the capture window
    2. Fetches odds from The Odds API (with Pinnacle)
    3. Saves closing odds to the database
    4. Updates any pending bets with CLV data

    Usage:
        collector = ClosingOddsCollector()
        captured = collector.capture()
        # Returns number of matches for which closing odds were captured
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        odds_client: Optional[OddsAPIClient] = None,
        capture_window_minutes: int = CAPTURE_WINDOW_MINUTES,
    ):
        self.db = db or Database()
        self.odds_client = odds_client or OddsAPIClient()
        self.capture_window = capture_window_minutes

    def capture(self) -> int:
        """
        Main entry point. Capture closing odds for imminent matches.

        Returns:
            Number of matches for which closing odds were captured.
        """
        now = datetime.now(timezone.utc)
        captured_count = 0

        # For each league, check if any matches are about to start
        for league, sport_key in OddsAPIClient.SPORT_KEYS.items():
            if league == League.NBA:
                continue  # Skip non-football leagues

            try:
                closing_data = self._capture_league(league, sport_key, now)
                for data in closing_data:
                    self._save_closing_odds(data)
                    self._update_pending_bets(data)
                    captured_count += 1

            except Exception as e:
                logger.error(f"Failed to capture closing odds for {league.value}: {e}")

        if captured_count > 0:
            logger.info(f"Captured closing odds for {captured_count} matches")

        return captured_count

    def _capture_league(
        self,
        league: League,
        sport_key: str,
        now: datetime,
    ) -> List[ClosingOddsData]:
        """Fetch and process closing odds for one league."""
        # Fetch odds — no cache so we get the freshest possible data
        response = self.odds_client.get_odds(sport_key)
        if not response.success:
            logger.warning(f"API error for {league.value}: {response.error}")
            return []

        # Clear cache for this request so next call gets fresh data
        self.odds_client._cache.clear()

        results = []

        for event in response.data or []:
            commence_time = datetime.fromisoformat(
                event["commence_time"].replace("Z", "+00:00")
            )

            # Only capture for matches starting within the window
            time_to_kickoff = (commence_time - now).total_seconds() / 60
            if time_to_kickoff < 0 or time_to_kickoff > self.capture_window:
                continue

            match_id = event["id"]

            # Check if we already captured closing odds for this match
            existing = self.db.get_closing_odds(match_id)
            if existing:
                # Only overwrite if this capture is closer to kickoff
                existing_mins = existing.get("minutes_before_kickoff", 999)
                if time_to_kickoff >= existing_mins:
                    continue

            # Parse all bookmaker odds for this event
            snapshots = self._parse_event_odds(event)
            if not snapshots:
                continue

            closing = ClosingOddsData(
                match_id=match_id,
                captured_at=now,
                minutes_before_kickoff=int(time_to_kickoff),
                snapshots=snapshots,
            )

            home_team = event.get("home_team", "?")
            away_team = event.get("away_team", "?")
            pinnacle_str = (
                f" (Pinnacle: {closing.pinnacle_home}/{closing.pinnacle_draw}/{closing.pinnacle_away})"
                if closing.pinnacle_home
                else " (no Pinnacle)"
            )
            logger.info(
                f"Closing odds: {home_team} vs {away_team} "
                f"[{int(time_to_kickoff)}min to KO]{pinnacle_str}"
            )

            results.append(closing)

        return results

    def _parse_event_odds(self, event: dict) -> List[OddsSnapshot]:
        """Parse all bookmaker odds from an API event."""
        snapshots = []

        for bookmaker in event.get("bookmakers", []):
            try:
                h2h_market = None
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        h2h_market = market
                        break

                if not h2h_market:
                    continue

                home_odds = None
                draw_odds = None
                away_odds = None

                for outcome in h2h_market["outcomes"]:
                    if outcome["name"] == event["home_team"]:
                        home_odds = outcome["price"]
                    elif outcome["name"] == event["away_team"]:
                        away_odds = outcome["price"]
                    elif outcome["name"] == "Draw":
                        draw_odds = outcome["price"]

                if home_odds and away_odds:
                    snapshots.append(
                        OddsSnapshot(
                            match_id=event["id"],
                            bookmaker=bookmaker["key"],
                            timestamp=datetime.fromisoformat(
                                bookmaker["last_update"].replace("Z", "+00:00")
                            ),
                            home_odds=home_odds,
                            draw_odds=draw_odds,
                            away_odds=away_odds,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to parse odds from {bookmaker.get('key', '?')}: {e}")

        return snapshots

    def _save_closing_odds(self, data: ClosingOddsData) -> None:
        """Save closing odds to the database."""
        self.db.save_closing_odds(
            match_id=data.match_id,
            captured_at=data.captured_at,
            minutes_before_kickoff=data.minutes_before_kickoff,
            pinnacle_home=data.pinnacle_home,
            pinnacle_draw=data.pinnacle_draw,
            pinnacle_away=data.pinnacle_away,
            avg_home=data.avg_home,
            avg_draw=data.avg_draw,
            avg_away=data.avg_away,
            best_home=data.best_home,
            best_draw=data.best_draw,
            best_away=data.best_away,
        )

    def _update_pending_bets(self, data: ClosingOddsData) -> None:
        """Update pending bets with closing odds for CLV calculation."""
        pending_bets = self.db.get_pending_bets()

        for bet in pending_bets:
            if bet.match_id != data.match_id:
                continue

            closing_odds_value = data.get_closing_odds_for_outcome(bet.outcome)
            if closing_odds_value:
                clv_pct = (bet.odds_at_placement / closing_odds_value - 1) * 100
                logger.info(
                    f"CLV for bet {bet.id}: placed at {bet.odds_at_placement}, "
                    f"closing at {closing_odds_value}, CLV = {clv_pct:+.1f}%"
                )

                # Store closing odds on the bet (will be fully settled later
                # when result comes in, but CLV is known as soon as closing line is captured)
                self.db.update_bet_closing_odds(
                    bet_id=bet.id,
                    closing_odds=closing_odds_value,
                    clv_pct=clv_pct,
                )

        # Also save each individual odds snapshot to the odds_snapshots table
        # for line movement analysis
        for snapshot in data.snapshots:
            self.db.save_odds_snapshot(snapshot)
