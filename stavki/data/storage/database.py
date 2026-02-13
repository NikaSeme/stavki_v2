"""
SQLite database storage for STAVKI.

Provides persistent storage for:
- Historical matches and results
- Odds snapshots (for CLV tracking)
- Placed bets and P&L
- Feature cache
- Model predictions

Uses SQLite for simplicity and portability.
All tables are auto-created on first run.
"""

import sqlite3
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager
import logging

from ..schemas import (
    Match, OddsSnapshot, BestOdds, MatchResult,
    PlacedBet, BetStatus, Prediction, DailyStats,
    League, Outcome, Team
)

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database for persistent storage.
    
    Thread-safe with connection per operation.
    Auto-creates tables on initialization.
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str = "artifacts/stavki.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_db()
    
    @contextmanager
    def _connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            # Version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            """)
            
            # Matches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    league TEXT NOT NULL,
                    commence_time TEXT NOT NULL,
                    home_score INTEGER,
                    away_score INTEGER,
                    season TEXT,
                    matchday INTEGER,
                    source TEXT,
                    source_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Odds snapshots (for line movement tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS odds_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    bookmaker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    home_odds REAL NOT NULL,
                    draw_odds REAL,
                    away_odds REAL NOT NULL,
                    market TEXT DEFAULT 'h2h',
                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)
            
            # Best odds per match (computed)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS best_odds (
                    match_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    home_odds REAL NOT NULL,
                    home_bookmaker TEXT,
                    draw_odds REAL,
                    draw_bookmaker TEXT,
                    away_odds REAL NOT NULL,
                    away_bookmaker TEXT,
                    home_book_count INTEGER,
                    draw_book_count INTEGER,
                    away_book_count INTEGER,
                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)
            
            # Closing odds (for CLV tracking)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS closing_odds (
                    match_id TEXT PRIMARY KEY,
                    captured_at TEXT NOT NULL,
                    minutes_before_kickoff INTEGER,
                    pinnacle_home REAL,
                    pinnacle_draw REAL,
                    pinnacle_away REAL,
                    avg_home REAL NOT NULL,
                    avg_draw REAL,
                    avg_away REAL NOT NULL,
                    best_home REAL NOT NULL,
                    best_draw REAL,
                    best_away REAL NOT NULL,
                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)
            
            # Predictions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    poisson_probs TEXT NOT NULL,
                    catboost_probs TEXT NOT NULL,
                    neural_probs TEXT NOT NULL,
                    ensemble_probs TEXT NOT NULL,
                    ensemble_weights TEXT,
                    disagreement_score REAL,
                    confidence_score REAL,
                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)
            
            # Placed bets
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bets (
                    id TEXT PRIMARY KEY,
                    match_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    odds_at_placement REAL NOT NULL,
                    stake REAL NOT NULL,
                    bookmaker TEXT NOT NULL,
                    placed_at TEXT NOT NULL,
                    model_prob REAL NOT NULL,
                    ev_at_placement REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    settled_at TEXT,
                    result TEXT,
                    profit_loss REAL,
                    closing_odds REAL,
                    clv_pct REAL,
                    FOREIGN KEY (match_id) REFERENCES matches(id)
                )
            """)
            
            # Daily statistics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date TEXT PRIMARY KEY,
                    total_bets INTEGER DEFAULT 0,
                    total_staked REAL DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    voids INTEGER DEFAULT 0,
                    gross_profit REAL DEFAULT 0,
                    gross_loss REAL DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    roi_pct REAL DEFAULT 0,
                    avg_odds REAL DEFAULT 0,
                    avg_ev REAL DEFAULT 0,
                    avg_clv REAL DEFAULT 0
                )
            """)
            
            # Create indexes for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_commence_time ON matches(commence_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_odds_match_id ON odds_snapshots(match_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_bets_match_id ON bets(match_id)
            """)
            
            # Record schema version
            cursor.execute("""
                INSERT OR IGNORE INTO schema_version (version, applied_at) 
                VALUES (?, ?)
            """, (self.SCHEMA_VERSION, datetime.utcnow().isoformat()))
            
            logger.info(f"Database initialized: {self.db_path}")
    
    # ==================== MATCHES ====================
    
    def save_match(self, match: Match) -> None:
        """Save or update a match."""
        with self._connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow().isoformat()
            
            cursor.execute("""
                INSERT OR REPLACE INTO matches 
                (id, home_team, away_team, league, commence_time, 
                 home_score, away_score, season, matchday, source, source_id,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        COALESCE((SELECT created_at FROM matches WHERE id = ?), ?), ?)
            """, (
                match.id,
                match.home_team.normalized_name,
                match.away_team.normalized_name,
                match.league.value,
                match.commence_time.isoformat(),
                match.home_score,
                match.away_score,
                match.season,
                match.matchday,
                match.source,
                match.source_id,
                match.id, now, now,
            ))
    
    def save_matches(self, matches: List[Match]) -> int:
        """Save multiple matches. Returns count saved."""
        for match in matches:
            self.save_match(match)
        return len(matches)
    
    def get_match(self, match_id: str) -> Optional[Match]:
        """Get match by ID."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches WHERE id = ?", (match_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_match(row)
            return None
    
    def get_upcoming_matches(
        self,
        league: Optional[League] = None,
        hours_ahead: int = 48
    ) -> List[Match]:
        """Get upcoming matches."""
        with self._connection() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow()
            cutoff = now.isoformat()
            
            if league:
                cursor.execute("""
                    SELECT * FROM matches 
                    WHERE league = ? AND commence_time > ? 
                    ORDER BY commence_time
                """, (league.value, cutoff))
            else:
                cursor.execute("""
                    SELECT * FROM matches 
                    WHERE commence_time > ? 
                    ORDER BY commence_time
                """, (cutoff,))
            
            return [self._row_to_match(row) for row in cursor.fetchall()]
    
    def get_historical_matches(
        self,
        league: Optional[League] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> List[Match]:
        """Get historical matches with results."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM matches WHERE home_score IS NOT NULL"
            params = []
            
            if league:
                query += " AND league = ?"
                params.append(league.value)
            
            if start_date:
                query += " AND commence_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND commence_time <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY commence_time"
            cursor.execute(query, params)
            
            return [self._row_to_match(row) for row in cursor.fetchall()]
    
    def update_match_result(self, match_id: str, home_score: int, away_score: int) -> None:
        """Update match with final result."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE matches SET home_score = ?, away_score = ?, updated_at = ?
                WHERE id = ?
            """, (home_score, away_score, datetime.utcnow().isoformat(), match_id))
    
    def _row_to_match(self, row: sqlite3.Row) -> Match:
        """Convert database row to Match object."""
        return Match(
            id=row["id"],
            home_team=Team(
                name=row["home_team"],
                normalized_name=row["home_team"],
            ),
            away_team=Team(
                name=row["away_team"],
                normalized_name=row["away_team"],
            ),
            league=League(row["league"]),
            commence_time=datetime.fromisoformat(row["commence_time"]),
            home_score=row["home_score"],
            away_score=row["away_score"],
            season=row["season"],
            matchday=row["matchday"],
            source=row["source"] or "unknown",
            source_id=row["source_id"],
        )
    
    # ==================== ODDS ====================
    
    def save_odds_snapshot(self, odds: OddsSnapshot) -> None:
        """Save an odds snapshot for line tracking."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO odds_snapshots 
                (match_id, bookmaker, timestamp, home_odds, draw_odds, away_odds, market)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                odds.match_id,
                odds.bookmaker,
                odds.timestamp.isoformat(),
                odds.home_odds,
                odds.draw_odds,
                odds.away_odds,
                odds.market,
            ))
    
    def save_best_odds(self, best_odds: BestOdds) -> None:
        """Save best odds for a match."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO best_odds 
                (match_id, timestamp, home_odds, home_bookmaker, draw_odds, draw_bookmaker,
                 away_odds, away_bookmaker, home_book_count, draw_book_count, away_book_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                best_odds.match_id,
                best_odds.timestamp.isoformat(),
                best_odds.home_odds,
                best_odds.home_bookmaker,
                best_odds.draw_odds,
                best_odds.draw_bookmaker,
                best_odds.away_odds,
                best_odds.away_bookmaker,
                best_odds.home_book_count,
                best_odds.draw_book_count,
                best_odds.away_book_count,
            ))
    
    def get_odds_history(self, match_id: str) -> List[OddsSnapshot]:
        """Get all odds snapshots for a match (for line movement analysis)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM odds_snapshots 
                WHERE match_id = ? 
                ORDER BY timestamp
            """, (match_id,))
            
            return [
                OddsSnapshot(
                    match_id=row["match_id"],
                    bookmaker=row["bookmaker"],
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    home_odds=row["home_odds"],
                    draw_odds=row["draw_odds"],
                    away_odds=row["away_odds"],
                    market=row["market"],
                )
                for row in cursor.fetchall()
            ]
    
    def save_closing_odds(
        self,
        match_id: str,
        captured_at: datetime,
        minutes_before_kickoff: int,
        pinnacle_home: Optional[float] = None,
        pinnacle_draw: Optional[float] = None,
        pinnacle_away: Optional[float] = None,
        avg_home: float = 0.0,
        avg_draw: Optional[float] = None,
        avg_away: float = 0.0,
        best_home: float = 0.0,
        best_draw: Optional[float] = None,
        best_away: float = 0.0,
    ) -> None:
        """Save closing odds for a match (for CLV tracking)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO closing_odds 
                (match_id, captured_at, minutes_before_kickoff,
                 pinnacle_home, pinnacle_draw, pinnacle_away,
                 avg_home, avg_draw, avg_away,
                 best_home, best_draw, best_away)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                captured_at.isoformat(),
                minutes_before_kickoff,
                pinnacle_home,
                pinnacle_draw,
                pinnacle_away,
                avg_home,
                avg_draw,
                avg_away,
                best_home,
                best_draw,
                best_away,
            ))
            logger.debug(f"Saved closing odds for match {match_id}")
    
    def get_closing_odds(self, match_id: str) -> Optional[Dict[str, Any]]:
        """
        Get closing odds for a match.
        
        Returns:
            Dict with keys: pinnacle_home, pinnacle_draw, pinnacle_away,
            avg_home, avg_draw, avg_away, best_home, best_draw, best_away,
            minutes_before_kickoff, captured_at. None if not found.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM closing_odds WHERE match_id = ?",
                (match_id,),
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "match_id": row["match_id"],
                "captured_at": row["captured_at"],
                "minutes_before_kickoff": row["minutes_before_kickoff"],
                "pinnacle_home": row["pinnacle_home"],
                "pinnacle_draw": row["pinnacle_draw"],
                "pinnacle_away": row["pinnacle_away"],
                "avg_home": row["avg_home"],
                "avg_draw": row["avg_draw"],
                "avg_away": row["avg_away"],
                "best_home": row["best_home"],
                "best_draw": row["best_draw"],
                "best_away": row["best_away"],
            }
    
    def update_bet_closing_odds(
        self,
        bet_id: str,
        closing_odds: float,
        clv_pct: float,
    ) -> None:
        """
        Update a bet's closing odds and CLV without settling it.
        
        This is called when closing odds are captured (before kickoff),
        separate from settlement (after the match ends).
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE bets SET 
                    closing_odds = ?,
                    clv_pct = ?
                WHERE id = ?
            """, (closing_odds, clv_pct, bet_id))
            logger.debug(f"Updated CLV for bet {bet_id}: {clv_pct:+.1f}%")
    

    # ==================== BETS ====================
    
    def save_bet(self, bet: PlacedBet) -> None:
        """Save a placed bet."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO bets 
                (id, match_id, outcome, odds_at_placement, stake, bookmaker, placed_at,
                 model_prob, ev_at_placement, status, settled_at, result, profit_loss,
                 closing_odds, clv_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                bet.id,
                bet.match_id,
                bet.outcome,
                bet.odds_at_placement,
                bet.stake,
                bet.bookmaker,
                bet.placed_at.isoformat(),
                bet.model_prob,
                bet.ev_at_placement,
                bet.status.value,
                bet.settled_at.isoformat() if bet.settled_at else None,
                bet.result,
                bet.profit_loss,
                bet.closing_odds,
                bet.clv_pct,
            ))
    
    def get_pending_bets(self) -> List[PlacedBet]:
        """Get all unsettled bets."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM bets WHERE status = 'pending'
            """)
            return [self._row_to_bet(row) for row in cursor.fetchall()]
    
    def get_bets_for_date_range(
        self,
        start_date: date,
        end_date: date
    ) -> List[PlacedBet]:
        """Get bets for a date range."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM bets 
                WHERE date(placed_at) BETWEEN ? AND ?
                ORDER BY placed_at
            """, (start_date.isoformat(), end_date.isoformat()))
            return [self._row_to_bet(row) for row in cursor.fetchall()]
    
    def update_bet_status(
        self,
        bet_id: str,
        status: BetStatus,
        result: Optional[str] = None,
        profit_loss: Optional[float] = None,
        closing_odds: Optional[float] = None
    ) -> None:
        """Update bet status after settlement."""
        with self._connection() as conn:
            cursor = conn.cursor()
            
            clv_pct = None
            if closing_odds:
                # Calculate CLV
                cursor.execute("SELECT odds_at_placement FROM bets WHERE id = ?", (bet_id,))
                row = cursor.fetchone()
                if row:
                    clv_pct = (row["odds_at_placement"] / closing_odds - 1) * 100
            
            cursor.execute("""
                UPDATE bets SET 
                    status = ?, 
                    settled_at = ?, 
                    result = ?,
                    profit_loss = ?,
                    closing_odds = ?,
                    clv_pct = ?
                WHERE id = ?
            """, (
                status.value,
                datetime.utcnow().isoformat(),
                result,
                profit_loss,
                closing_odds,
                clv_pct,
                bet_id,
            ))
    
    def _row_to_bet(self, row: sqlite3.Row) -> PlacedBet:
        """Convert database row to PlacedBet."""
        return PlacedBet(
            id=row["id"],
            match_id=row["match_id"],
            outcome=row["outcome"],
            odds_at_placement=row["odds_at_placement"],
            stake=row["stake"],
            bookmaker=row["bookmaker"],
            placed_at=datetime.fromisoformat(row["placed_at"]),
            model_prob=row["model_prob"],
            ev_at_placement=row["ev_at_placement"],
            status=BetStatus(row["status"]),
            settled_at=datetime.fromisoformat(row["settled_at"]) if row["settled_at"] else None,
            result=row["result"],
            profit_loss=row["profit_loss"],
            closing_odds=row["closing_odds"],
            clv_pct=row["clv_pct"],
        )
    
    # ==================== STATISTICS ====================
    
    def get_daily_stats(self, dt: date) -> Optional[DailyStats]:
        """Get statistics for a specific day."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_stats WHERE date = ?", (dt.isoformat(),))
            row = cursor.fetchone()
            
            if row:
                return DailyStats(
                    date=row["date"],
                    total_bets=row["total_bets"],
                    total_staked=row["total_staked"],
                    wins=row["wins"],
                    losses=row["losses"],
                    voids=row["voids"],
                    gross_profit=row["gross_profit"],
                    gross_loss=row["gross_loss"],
                    net_pnl=row["net_pnl"],
                    roi_pct=row["roi_pct"],
                    avg_odds=row["avg_odds"],
                    avg_ev=row["avg_ev"],
                    avg_clv=row["avg_clv"],
                )
            return None
    
    def compute_daily_stats(self, dt: date) -> DailyStats:
        """Compute and save daily statistics from bets."""
        bets = self.get_bets_for_date_range(dt, dt)
        
        stats = DailyStats(date=dt.isoformat())
        
        for bet in bets:
            stats.total_bets += 1
            stats.total_staked += bet.stake
            stats.avg_odds = (stats.avg_odds * (stats.total_bets - 1) + bet.odds_at_placement) / stats.total_bets
            stats.avg_ev = (stats.avg_ev * (stats.total_bets - 1) + bet.ev_at_placement) / stats.total_bets
            
            if bet.clv_pct:
                stats.avg_clv = (stats.avg_clv * (stats.total_bets - 1) + bet.clv_pct) / stats.total_bets
            
            if bet.status == BetStatus.WON:
                stats.wins += 1
                stats.gross_profit += bet.profit_loss or 0
            elif bet.status == BetStatus.LOST:
                stats.losses += 1
                stats.gross_loss += abs(bet.profit_loss or 0)
            elif bet.status == BetStatus.VOID:
                stats.voids += 1
        
        stats.net_pnl = stats.gross_profit - stats.gross_loss
        if stats.total_staked > 0:
            stats.roi_pct = stats.net_pnl / stats.total_staked * 100
        
        # Save to database
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_stats VALUES 
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats.date,
                stats.total_bets,
                stats.total_staked,
                stats.wins,
                stats.losses,
                stats.voids,
                stats.gross_profit,
                stats.gross_loss,
                stats.net_pnl,
                stats.roi_pct,
                stats.avg_odds,
                stats.avg_ev,
                stats.avg_clv,
            ))
        
        return stats
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get overall betting statistics."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_bets,
                    SUM(stake) as total_staked,
                    SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status = 'lost' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN profit_loss < 0 THEN ABS(profit_loss) ELSE 0 END) as gross_loss,
                    SUM(COALESCE(profit_loss, 0)) as net_pnl,
                    AVG(odds_at_placement) as avg_odds,
                    AVG(ev_at_placement) as avg_ev,
                    AVG(clv_pct) as avg_clv
                FROM bets
            """)
            row = cursor.fetchone()
            
            total_staked = row["total_staked"] or 0
            net_pnl = row["net_pnl"] or 0
            
            return {
                "total_bets": row["total_bets"] or 0,
                "total_staked": total_staked,
                "wins": row["wins"] or 0,
                "losses": row["losses"] or 0,
                "gross_profit": row["gross_profit"] or 0,
                "gross_loss": row["gross_loss"] or 0,
                "net_pnl": net_pnl,
                "roi_pct": (net_pnl / total_staked * 100) if total_staked > 0 else 0,
                "avg_odds": row["avg_odds"] or 0,
                "avg_ev": row["avg_ev"] or 0,
                "avg_clv": row["avg_clv"] or 0,
            }
