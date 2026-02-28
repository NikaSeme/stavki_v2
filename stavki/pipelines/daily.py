"""
Daily Betting Pipeline
======================

Main orchestration pipeline for daily value bet discovery:
1. Collect fresh odds data
2. Build features for upcoming matches
3. Run ensemble model predictions
4. Calculate EV and value bets
5. Apply filters and staking
6. Output actionable bets

Usage:
    pipeline = DailyPipeline()
    bets = pipeline.run(sport="soccer_epl")
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

from stavki.config import get_config
from stavki.data.collectors.sportmonks import SportMonksClient
from stavki.data.schemas.match import (
    MatchEnrichment, RefereeInfo, WeatherInfo, InjuryInfo,
    CoachInfo, VenueInfo,
)

logger = logging.getLogger(__name__)

# Prompt 2: Critical features for 1x2 prediction path.
# If any of these are missing or NaN per row, the match is marked _invalid_for_bet.
CRITICAL_FEATURES_1X2 = [
    "elo_home", "elo_away",
    "form_home_pts", "form_away_pts",
    "imp_home_norm", "imp_draw_norm", "imp_away_norm",
]


@dataclass
class PipelineConfig:
    """Configuration for daily pipeline."""
    # Data
    leagues: List[str] = field(default_factory=lambda: [
        "soccer_epl", 
        "soccer_spain_la_liga", 
        "soccer_germany_bundesliga", 
        "soccer_italy_serie_a", 
        "soccer_france_ligue_one",
        "soccer_efl_champ",
    ])
    max_matches: int = 50
    scan_window_hours: int = 72  # Lookahead window (3 days)
    
    # Model
    use_ensemble: bool = True
    models: List[str] = field(default_factory=lambda: ["catboost", "poisson", "neural"])
    
    # Strategy
    min_ev: float = 0.03
    max_stake_pct: float = 0.05
    # BOB'S AGGRESSIVE STRATEGY: 0.75 KELLY
    kelly_fraction: float = 0.75
    
    # Filters
    min_confidence: float = 0.05
    max_divergence: float = 0.25
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    save_predictions: bool = True


@dataclass
class BetCandidate:
    """A potential value bet."""
    match_id: str
    home_team: str
    away_team: str
    league: str
    kickoff: datetime
    market: str
    selection: str
    odds: float
    bookmaker: str
    
    # Model outputs
    model_prob: float
    market_prob: float
    blended_prob: float
    
    # Strategy
    ev: float
    edge: float
    stake_pct: float
    stake_amount: float
    
    # Quality
    confidence: float
    justified_score: int
    divergence_level: str
    
    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "match": f"{self.home_team} vs {self.away_team}",
            "league": self.league,
            "kickoff": self.kickoff.isoformat() if self.kickoff else None,
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "bookmaker": self.bookmaker,
            "model_prob": round(self.model_prob, 4),
            "market_prob": round(self.market_prob, 4),
            "ev": round(self.ev, 4),
            "edge": round(self.edge, 4),
            "stake_pct": round(self.stake_pct, 4),
            "stake_amount": round(self.stake_amount, 2),
            "confidence": round(self.confidence, 4),
            "justified_score": self.justified_score,
        }


class DailyPipeline:
    """
    Main pipeline for daily betting operations.
    
    Orchestrates: Data → Features → Models → Strategy → Output
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        bankroll: float = 1000.0,
    ):
        self.config = config or PipelineConfig()
        self.bankroll = bankroll
        
        # Initialize components (lazy loading)
        self._data_collector = None
        self._feature_builder = None
        self._models = None
        self._staker = None
        self._filters = None
        self._blender = None
        self._router = None
        self._calibrator = None
    
    def _init_components(self):
        """Initialize all pipeline components."""
        from stavki.strategy import (
            EVCalculator, KellyStaker, BetFilters,
            LeagueRouter, LiquidityBlender,
        )
        
        if self._router is None:
            # Use optimized league weights
            root_dir = Path(__file__).resolve().parents[2]
            config_path = root_dir / "stavki" / "config" / "leagues.json"
            self._router = LeagueRouter(config_path=config_path)
        
        if self._blender is None:
            self._blender = LiquidityBlender(league_router=self._router)
        
        if self._staker is None:
            # Reconstruct path to data/kelly_state.json relative to this file
            # daily.py -> pipelines -> stavki -> [root] -> data
            root_dir = Path(__file__).resolve().parents[2]
            state_path = root_dir / "data" / "kelly_state.json"
            
            self._staker = KellyStaker(
                bankroll=self.bankroll,
                config={
                    "kelly_fraction": self.config.kelly_fraction,
                    "min_stake_amount": 0.1,  # Allow 10 cent bets to verify system works
                    "min_stake_pct": 0.0001,  # 0.01%
                },
                state_file=state_path
            )
        
        if self._filters is None:
            self._filters = BetFilters(config={
                "min_ev": self.config.min_ev,
                "min_confidence": self.config.min_confidence,
            })
        
        self._ev_calc = EVCalculator()
        
        # Load calibrator if available
        self._load_calibrator()
    
    def run(
        self,
        odds_df: Optional[pd.DataFrame] = None,
        matches_df: Optional[pd.DataFrame] = None,
        model_probs: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    ) -> List[BetCandidate]:
        """
        Execute the full daily pipeline.
        
        Args:
            odds_df: Pre-loaded odds data (optional, will fetch if None)
            matches_df: Match details (optional)
            model_probs: Pre-computed model probabilities (optional)
        
        Returns:
            List of BetCandidate objects ranked by value
        """
        logger.info("=" * 50)
        logger.info("Starting Daily Pipeline")
        logger.info("=" * 50)
        
        self._init_components()
        
        # Step 1: Load or fetch odds
        if odds_df is None:
            logger.info("Step 1: Fetching odds data...")
            odds_df = self._fetch_odds()
        else:
            logger.info("Step 1: Using provided odds data")
        
        if odds_df is None or odds_df.empty:
            logger.warning("No odds data available")
            return []
        
        logger.info(f"  → {len(odds_df)} odds rows loaded")
        
        # Step 2: Extract unique matches
        if matches_df is None:
            matches_df = self._extract_matches(odds_df)
        
        logger.info(f"Step 2: {len(matches_df)} unique matches")
        
        # Step 2.5: Enrich matches with SportMonks data (weather, injuries, etc.)
        logger.info("Step 2.5: Enriching matches with SportMonks data...")
        self._enrich_matches(matches_df)
        
        # Step 3: Build features (if features module available)
        logger.info("Step 3: Building features...")
        features_df = self._build_features(matches_df, odds_df)
        
        # ── Prompt 2 gap fix: Propagate _invalid_for_bet into matches_df ──
        # features_df (returned by _build_features → _map_features_to_model_inputs)
        # may contain _invalid_for_bet. Merge it into matches_df so _find_value_bets sees it.
        if "_invalid_for_bet" in features_df.columns:
            id_col_m = "event_id" if "event_id" in matches_df.columns else "match_id"
            id_col_f = "event_id" if "event_id" in features_df.columns else "match_id"
            
            if id_col_f in features_df.columns:
                inv_flags = features_df[[id_col_f, "_invalid_for_bet"]].copy()
                inv_flags[id_col_f] = inv_flags[id_col_f].astype(str)
                
                if "_invalid_for_bet" in matches_df.columns:
                    matches_df = matches_df.drop(columns=["_invalid_for_bet"])
                
                matches_df[id_col_m] = matches_df[id_col_m].astype(str)
                matches_df = matches_df.merge(
                    inv_flags.rename(columns={id_col_f: id_col_m}),
                    on=id_col_m,
                    how="left",
                )
                matches_df["_invalid_for_bet"] = matches_df["_invalid_for_bet"].fillna(False)
                
                n_invalid = matches_df["_invalid_for_bet"].sum()
                if n_invalid > 0:
                    logger.warning(
                        f"🚫 Prompt 2: {n_invalid}/{len(matches_df)} matches will be "
                        f"excluded from value bets (invalid_for_bet)"
                    )
        
        # Step 4: Get model predictions
        if model_probs is None:
            logger.info("Step 4: Running model predictions...")
            model_probs = self._get_predictions(matches_df, features_df)
        else:
            logger.info("Step 4: Using provided predictions")
        
        # Step 5: Get best odds per outcome
        logger.info("Step 5: Selecting best prices...")
        best_prices = self._select_best_prices(odds_df)
        
        # Step 6: Calculate market probabilities (no-vig)
        logger.info("Step 6: Computing market probabilities...")
        market_probs = self._compute_market_probs(best_prices)
        
        # Step 7: Find value bets
        logger.info("Step 7: Finding value bets...")
        candidates = self._find_value_bets(
            matches_df, model_probs, market_probs, best_prices
        )
        
        logger.info(f"  → {len(candidates)} value bets found")
        
        # Step 8: Apply filters
        logger.info("Step 8: Applying filters...")
        filtered = self._apply_filters(candidates)
        logger.info(f"  → {len(filtered)} bets passed filters")
        
        # Step 9: Calculate stakes
        logger.info("Step 9: Calculating stakes...")
        final_bets = self._calculate_stakes(filtered)
        
        # Step 10: Save output
        if self.config.save_predictions and final_bets:
            self._save_output(final_bets)
        
        logger.info("=" * 50)
        logger.info(f"Pipeline complete: {len(final_bets)} actionable bets")
        logger.info("=" * 50)
        
        return final_bets
    
    def _fetch_odds(self) -> Optional[pd.DataFrame]:
        """Fetch fresh odds from API or load from cached file.
        
        Strategy:
        1. Check for recent cached CSV (< 30 min old)
        2. If stale or missing, fetch live from OddsAPI
        3. Cache the result as timestamped CSV
        4. Fall back to stale cache if API fails
        """
        odds_dir = self.config.output_dir / "odds"
        odds_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Check for recent cache
        cached_df = None
        if odds_dir.exists():
            files = sorted(odds_dir.glob("*.csv"))
            if files:
                latest = files[-1]
                import os
                age_minutes = (datetime.now().timestamp() - os.path.getmtime(latest)) / 60
                cached_df = pd.read_csv(latest)
                if age_minutes < 5:
                    logger.info(f"Using fresh cached odds ({age_minutes:.1f}m old): {latest.name}")
                    return cached_df
                logger.info(f"Cache is {age_minutes:.1f}m old (limit 5m), attempting live fetch...")
        
        # 2. Fetch live odds via Collectors
        try:
            from stavki.data.collectors import (
                OddsAPICollector, OddsAPIClient,
                SportMonksCollector, BetfairCollector
            )
            from stavki.data.schemas import League
            from stavki.config import get_config
            from stavki.data.processors.validate import OddsValidator
            
            config = get_config()
            
            # Initialize collectors
            sm_collector = None
            if config.sportmonks_api_key:
                sm_collector = SportMonksCollector()
            
            bf_collector = None
            if config.betfair_app_key:
                bf_collector = BetfairCollector()
                
            oa_collector = None
            if config.odds_api_key:
                oa_collector = OddsAPICollector()

            # Map league strings to League enum keys
            league_map = {
                "soccer_epl": League.EPL,
                "soccer_germany_bundesliga": League.BUNDESLIGA, 
                "soccer_spain_la_liga": League.LA_LIGA,
                "soccer_italy_serie_a": League.SERIE_A,
                "soccer_france_ligue_one": League.LIGUE_1,
                "soccer_efl_champ": League.CHAMPIONSHIP,
            }
            
            rows = []
            
            for league_str in self.config.leagues:
                if league_str not in league_map:
                    logger.warning(f"Skipping unknown league: {league_str}")
                    continue
                    
                league = league_map[league_str]
                matches = []
                
                # A. Get Fixtures (Prefer SportMonks -> OddsAPI)
                if sm_collector:
                    try:
                        matches = sm_collector.fetch_matches(league, max_hours_ahead=self.config.scan_window_hours)
                        logger.info(f"Fetched {len(matches)} fixtures from SportMonks for {league.name} (next {self.config.scan_window_hours}h)")
                    except Exception as e:
                        logger.error(f"SportMonks fixture fetch failed for {league.name}: {e}")
                
                # Fallback to OddsAPI for fixtures if SportMonks returned nothing
                if not matches and oa_collector:
                    logger.info(f"Falling back to OddsAPI for {league.name} fixtures...")
                    try:
                        matches = oa_collector.fetch_matches(league, include_odds=True, max_hours_ahead=self.config.scan_window_hours)
                        logger.info(f"OddsAPI returned {len(matches)} fixtures for {league.name}")
                    except Exception as e:
                        logger.error(f"OddsAPI fixture fetch also failed for {league.name}: {e}")
                
                if not matches:
                    logger.info(f"No fixtures found for {league.name} in next {self.config.scan_window_hours}h")
                    continue

                # B. Get Odds (Prefer Betfair -> SportMonks -> OddsAPI)
                bf_odds = {}
                if bf_collector:
                    try:
                        bf_odds = bf_collector.fetch_odds(league, matches)
                        logger.info(f"Fetched Betfair odds for {len(bf_odds)}/ {len(matches)} matches")
                    except Exception as e:
                        logger.error(f"Betfair odds fetch failed: {e}")
                        
                sm_odds = {}
                if sm_collector and not bf_odds:
                    try:
                        sm_odds = sm_collector.fetch_odds(league, matches)
                        logger.info(f"Fetched SportMonks odds for {len(sm_odds)}/{len(matches)} {league.name} matches")
                    except Exception as e:
                        logger.error(f"SportMonks odds fetch failed for {league.name}: {e}")

                # C. Build Rows
                matches_with_odds = 0
                for m in matches:
                    row = {
                        "event_id": m.id,
                        "fixture_id": m.id,
                        "source_id": m.id,
                        "home_team": m.home_team.name,
                        "away_team": m.away_team.name,
                        "league": league.value,
                        "commence_time": m.commence_time.isoformat(),
                        "source": m.source
                    }
                    
                    primary_found = False
                    
                    # 1. SportMonks Odds (Multi-Market)
                    if m.id in sm_odds and sm_odds[m.id]:
                        for snap in sm_odds[m.id]:
                            if snap.market == "1x2":
                                if "home_odds" not in row:
                                    row.update({
                                        "home_odds": snap.home_odds,
                                        "draw_odds": snap.draw_odds,
                                        "away_odds": snap.away_odds,
                                        "home_bookmaker": snap.bookmaker,
                                        "away_bookmaker": snap.bookmaker,
                                    })
                                    primary_found = True
                            
                            elif snap.market == "corners_1x2":
                                row.update({
                                    "corners_home_odds": snap.home_odds,
                                    "corners_draw_odds": snap.draw_odds,
                                    "corners_away_odds": snap.away_odds,
                                })
                                
                            elif snap.market == "btts":
                                row.update({
                                    "btts_yes_odds": snap.home_odds,
                                    "btts_no_odds": snap.away_odds,
                                })

                    # 2. Betfair Fallback (for 1X2 only)
                    if m.id in bf_odds and bf_odds[m.id]:
                        best_snap = bf_odds[m.id][0]
                        row.update({
                            "home_odds": best_snap.home_odds,
                            "draw_odds": best_snap.draw_odds,
                            "away_odds": best_snap.away_odds,
                            "home_bookmaker": best_snap.bookmaker,
                            "away_bookmaker": best_snap.bookmaker,
                        })
                        primary_found = True

                    # 3. OddsAPI fallback for THIS specific match if no odds found
                    if not primary_found and oa_collector:
                        try:
                            sport_key = league_str
                            oa_resp = oa_collector.client.get_odds(sport_key)
                            if oa_resp.success and oa_resp.data:
                                for event in oa_resp.data:
                                    oa_home = event.get("home_team", "").lower()
                                    oa_away = event.get("away_team", "").lower()
                                    m_home = m.home_team.name.lower()
                                    m_away = m.away_team.name.lower()
                                    # Fuzzy match by checking if event teams contain match teams
                                    if (oa_home in m_home or m_home in oa_home) and (oa_away in m_away or m_away in oa_away):
                                        bookmakers = event.get("bookmakers", [])
                                        if bookmakers:
                                            bm = bookmakers[0]  # First bookmaker
                                            outcomes = {o["name"]: o["price"] for o in bm.get("markets", [{}])[0].get("outcomes", [])}
                                            if outcomes:
                                                row.update({
                                                    "home_odds": outcomes.get(m.home_team.name, outcomes.get("Home", 0)),
                                                    "draw_odds": outcomes.get("Draw", 0),
                                                    "away_odds": outcomes.get(m.away_team.name, outcomes.get("Away", 0)),
                                                    "home_bookmaker": bm.get("title", "OddsAPI"),
                                                    "away_bookmaker": bm.get("title", "OddsAPI"),
                                                })
                                                if row["home_odds"] > 1 and row["away_odds"] > 1:
                                                    primary_found = True
                                                    logger.info(f"  OddsAPI fallback provided odds for {m.home_team.name} vs {m.away_team.name}")
                                        break
                        except Exception as e:
                            logger.debug(f"OddsAPI per-match fallback failed: {e}")

                    if primary_found:
                        rows.append(row)
                        matches_with_odds += 1
                
                logger.info(f"  {league.name}: {matches_with_odds}/{len(matches)} matches have odds")
            
            if not rows:
                logger.warning("No matches with odds found across all sources")
                return cached_df
            
            df = pd.DataFrame(rows)
            
            # Save as timestamped cache
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cache_path = odds_dir / f"odds_{timestamp}.csv"
            df.to_csv(cache_path, index=False)
            logger.info(f"Saved {len(df)} matches to {cache_path.name}")
            
            return df
            
        except ImportError as e:
            logger.error(f"Collectors import failed: {e}")
            return cached_df
        except Exception as e:
            logger.error(f"Fetch failed: {e} — falling back to cache")
            import traceback
            traceback.print_exc()
            return cached_df
    
    def _extract_matches(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique matches from odds data."""
        if "event_id" not in odds_df.columns:
            # Create synthetic event_id
            odds_df["event_id"] = odds_df.apply(
                lambda r: f"{r.get('home_team', '')}_{r.get('away_team', '')}",
                axis=1
            )
        
        # Get unique matches
        match_cols = ["event_id", "home_team", "away_team"]
        optional_cols = ["league", "sport_key", "kickoff", "commence_time", "fixture_id", "source_id"]
        
        for col in optional_cols:
            if col in odds_df.columns:
                match_cols.append(col)
        
        matches = odds_df[match_cols].drop_duplicates("event_id")
        return matches.reset_index(drop=True)
    
    def _enrich_matches(self, matches_df: pd.DataFrame) -> None:
        """
        Enrich matches with SportMonks data for Tier 1-3 features.
        
        Fetches per-match: weather, injuries, referee, SM odds.
        Results are stored in a column 'enrichment_data' on the DataFrame.
        This data is later passed to the FeatureRegistry.
        """
        try:
            # Use os.getenv directly to avoid get_config() global lock/side-effects in threads
            import os
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            api_key = os.getenv("SPORTMONKS_API_KEY")
            
            if not api_key:
                logger.info("  → No SportMonks API key (env), skipping enrichment")
                return
            
            # Pass strict timeout/retries to avoid hangs
            # Default is 30s timeout, 5 retries. We want much faster fail.
            client = SportMonksClient(
                api_key=api_key,
                timeout=(3.05, 5),   # (connect, read) tuple for safety
                retries=1      # Max 1 retry
            )
        except Exception as e:
            logger.warning(f"  → SportMonks init failed: {e}, skipping enrichment")
            return
        
        enrichments = {}
        
        # Execute in chunks of 30 to stay well under API size limits (often 50 max)
        chunk_size = 30
        
        # 1. Map rows to Fixture IDs
        valid_rows = []
        for row in matches_df.to_dict('records'):
            match_id = row.get("event_id", f"{row.get('home_team', '')}_{row.get('away_team', '')}")
            fixture_id = row.get("fixture_id") or row.get("source_id")
            if fixture_id:
                 try:
                     f_id_int = int(fixture_id)
                     valid_rows.append((match_id, f_id_int))
                 except ValueError:
                     pass
        
        logger.info(f"Enriching {len(valid_rows)} matches via Batched API chunks...")
        
        # 2. Process chunks sequentially (1 API call per 30 matches)
        processed_count = 0
        for i in range(0, len(valid_rows), chunk_size):
            chunk = valid_rows[i:i + chunk_size]
            chunk_fids = [f_id for _, f_id in chunk]
            
            try:
                # One API call fetches everything for these 30 matches
                multi_data = client.get_multiple_fixtures_full(chunk_fids)
                
                # Assign back to Enrichments
                for match_id, f_id in chunk:
                    match_data = multi_data.get(f_id)
                    if not match_data:
                        continue
                        
                    enrichment = MatchEnrichment()
                    
                    # Participants
                    participants = match_data.get("participants", [])
                    home_tid = away_tid = None
                    for p in participants:
                        loc = p.get("meta", {}).get("location")
                        if loc == "home":
                            home_tid = p.get("id")
                        elif loc == "away":
                            away_tid = p.get("id")
                            
                    # Coaches
                    coaches = match_data.get("coaches", [])
                    for coach in coaches:
                        cid = coach.get("coach_id")
                        if coach.get("team_id") == home_tid:
                            enrichment.home_coach_id = cid
                        elif coach.get("team_id") == away_tid:
                            enrichment.away_coach_id = cid
                            
                    # Lineups
                    lineups = match_data.get("lineups", [])
                    h_players, a_players = [], []
                    h_pos, a_pos = [], []
                    if lineups:
                        for p in lineups:
                            if p.get("type_id") == 11:
                                pid = p.get("player_id")
                                pos_id = p.get("player", {}).get("position_id", 0) if isinstance(p.get("player"), dict) else 0
                                if pos_id is None: pos_id = 0
                                
                                if p.get("team_id") == home_tid:
                                    h_players.append(pid)
                                    h_pos.append(pos_id)
                                elif p.get("team_id") == away_tid:
                                    a_players.append(pid)
                                    a_pos.append(pos_id)
                                    
                    enrichment.home_player_ids = h_players
                    enrichment.away_player_ids = a_players
                    enrichment.home_positions = h_pos
                    enrichment.away_positions = a_pos
                    
                    # Weather
                    weather = match_data.get("weather", {})
                    if weather and isinstance(weather, dict):
                         enrichment.weather = WeatherInfo(
                             temperature_c=weather.get("temperature"),
                             wind_speed_ms=weather.get("wind"),
                             humidity_pct=weather.get("humidity"),
                             precipitation_mm=weather.get("precipitation"),
                             description=weather.get("description"),
                         )
                         
                    # Save
                    enrichments[match_id] = enrichment
                
                processed_count += len(chunk)
                logger.info(f"  → Enriched {processed_count}/{len(valid_rows)} matches via chunking...")
                
            except Exception as e:
                logger.warning(f"Batch enrichment failed for chunk start {i}: {e}")
        
        # Store enrichments on the DataFrame
        matches_df["_enrichment"] = matches_df.apply(
            lambda r: enrichments.get(
                r.get("event_id", f"{r.get('home_team', '')}_{r.get('away_team', '')}"),
                None
            ),
            axis=1
        )
        
        logger.info(f"  → {len(enrichments)}/{len(matches_df)} matches successfully enriched")
    
    def _build_features(
        self,
        matches_df: pd.DataFrame,
        odds_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build features for all matches using FeatureRegistry."""
        try:
            from stavki.features.registry import FeatureRegistry
            from stavki.data.schemas import Match, Team, League
            
            # 1. Load historical data for fitting feature builders
            hist_df = self._load_history()
            if hist_df.empty:
                logger.warning("No historical data found. Features will be limited.")
            
            # 2. Convert history to Match objects
            hist_matches = self._df_to_matches(hist_df, is_historical=True)
            
            # 3. Fit registry
            registry = FeatureRegistry()
            registry.fit(hist_matches)
            
            # 4. Convert current matches to Match objects
            current_matches = self._df_to_matches(matches_df, is_historical=False)
            
            # 5. Compute features
            features = registry.compute_batch(current_matches)
            
            # 6. Ensure index alignment
            if features.empty:
                features = pd.DataFrame(index=[m.id for m in current_matches])
            
            features = features.reset_index().rename(columns={"index": "match_id"})
            
            # --- Inject Odds Features ---
            # Compute real Avg/Max odds from ALL bookmakers, plus implied probabilities
            if not odds_df.empty and "home_odds" in odds_df.columns:
                odds_cols = ["event_id", "home_odds", "draw_odds", "away_odds"]
                odds_clean = odds_df[odds_cols].dropna(subset=["home_odds", "away_odds"])
                
                # Aggregate across bookmakers: mean → Avg, max → Max
                agg_odds = odds_clean.groupby("event_id").agg(
                    AvgH=("home_odds", "mean"),
                    AvgD=("draw_odds", "mean"),
                    AvgA=("away_odds", "mean"),
                    MaxH=("home_odds", "max"),
                    MaxD=("draw_odds", "max"),
                    MaxA=("away_odds", "max"),
                    n_bookmakers=("home_odds", "count"),
                ).reset_index().rename(columns={"event_id": "match_id"})
                
                # Compute implied probabilities from average odds
                agg_odds["imp_home"] = 1.0 / agg_odds["AvgH"]
                agg_odds["imp_draw"] = 1.0 / agg_odds["AvgD"].clip(lower=1.01)
                agg_odds["imp_away"] = 1.0 / agg_odds["AvgA"]
                agg_odds["margin"] = agg_odds["imp_home"] + agg_odds["imp_draw"] + agg_odds["imp_away"] - 1.0
                total_imp = agg_odds["imp_home"] + agg_odds["imp_draw"] + agg_odds["imp_away"]
                agg_odds["imp_home_norm"] = agg_odds["imp_home"] / total_imp
                agg_odds["imp_draw_norm"] = agg_odds["imp_draw"] / total_imp
                agg_odds["imp_away_norm"] = agg_odds["imp_away"] / total_imp
                
                # Also compute SportMonks-style implied columns
                agg_odds["sm_implied_home"] = agg_odds["imp_home_norm"]
                agg_odds["sm_implied_draw"] = agg_odds["imp_draw_norm"]
                agg_odds["sm_implied_away"] = agg_odds["imp_away_norm"]
                
                # REMOVED: Legacy bookmaker re-injection (B365, BW, IW, etc.) was contradicting
                # retrain_system.py which explicitly strips these columns during training.
                # Models trained without these features don't need them at inference time.

                # Drop helper column
                agg_odds = agg_odds.drop(columns=["n_bookmakers"])
                
                logger.info(f"  Computed odds aggregates from {len(odds_clean)} bookmaker entries → {len(agg_odds)} matches")
                
                # Ensure match_id type consistency (str)
                features["match_id"] = features["match_id"].astype(str)
                agg_odds["match_id"] = agg_odds["match_id"].astype(str)
                
                # Merge into features (don't overwrite existing columns)
                existing_cols = set(features.columns) - {"match_id"}
                new_cols = [c for c in agg_odds.columns if c not in existing_cols or c == "match_id"]
                features = pd.merge(features, agg_odds[new_cols], on="match_id", how="left")
            
            # --- Inject Live Arrays (Positions, Coaches, Lineups) ---
            if "_enrichment" in matches_df.columns:
                # Add default columns to features
                for col in ["home_coach_id", "away_coach_id"]:
                    if col not in features.columns:
                        features[col] = 0
                for col in ["home_player_ids", "away_player_ids", "home_positions", "away_positions"]:
                    if col not in features.columns:
                        features[col] = None
                        
                for idx, row in matches_df.iterrows():
                    mid = str(row.get("event_id", row.get("match_id")))
                    enrich = row.get("_enrichment")
                    if enrich:
                        idx_f = features.index[features["match_id"] == mid]
                        if len(idx_f) > 0:
                            if enrich.home_coach_id: features.at[idx_f[0], "home_coach_id"] = enrich.home_coach_id
                            if enrich.away_coach_id: features.at[idx_f[0], "away_coach_id"] = enrich.away_coach_id
                            
                            # Use at for lists to prevent pandas shape broadcasting issues!
                            features.at[idx_f[0], "home_player_ids"] = enrich.home_player_ids
                            features.at[idx_f[0], "away_player_ids"] = enrich.away_player_ids
                            features.at[idx_f[0], "home_positions"] = enrich.home_positions
                            features.at[idx_f[0], "away_positions"] = enrich.away_positions

            # 7. Map features to model expectations (Backwards Compatibility)
            features = self._map_features_to_model_inputs(features)
            
            # Merge
            # ensure matches_df has match_id or event_id
            id_col = "match_id" if "match_id" in matches_df.columns else "event_id"
            
            # Ensure merge keys are same type (string)
            matches_df[id_col] = matches_df[id_col].astype(str)
            features["match_id"] = features["match_id"].astype(str)
            
            merged = pd.merge(
                matches_df, 
                features, 
                left_on=id_col, 
                right_on="match_id", 
                how="left"
            )
            
            # --- Automation: Update Redis Memory Cache ---
            logger.info("Executing Live State Memory Serialization...")
            try:
                import os
                os.system("python3 scripts/cache_live_state.py")
            except Exception as e:
                logger.error(f"Failed to refresh Redis cache: {e}")

            # --- Validation ---
            # Check for critical features
            critical_cols = ["elo_home", "elo_away", "form_home_pts", "form_away_pts"]
            missing_stats = {}
            for col in critical_cols:
                if col in merged.columns:
                    missing = merged[col].isna().mean()
                    if missing > 0.20:
                        missing_stats[col] = missing
            
            if missing_stats:
                msg = f"Critical features missing > 20%: {missing_stats}"
                logger.error(msg)
                # We raise error to prevent garbage-in-garbage-out
                # But only if we have enough rows to matter
                if len(merged) > 5:
                    raise ValueError(msg)
            
            if "match_id_y" in merged.columns:
                merged = merged.drop(columns=["match_id_y"]).rename(columns={"match_id_x": "match_id"})
            
            return merged

        except Exception as e:
            logger.warning(f"Feature building failed: {e}. Using minimal features.")
            import traceback
            traceback.print_exc()
            # Return minimal features
            return matches_df.copy()

    def _map_features_to_model_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map FeatureRegistry output names to Training Data (features_full.csv) names.
        This ensures models trained on legacy/different names can still run.
        
        Prompt 2: Critical features are never filled with defaults.
        Non-critical features get smart defaults and substitution coverage is logged.
        """
        # 0. Inject CatBoost categorical features (HomeTeam/AwayTeam)
        if "HomeTeam" not in df.columns and "home_team" in df.columns:
            df["HomeTeam"] = df["home_team"]
        if "AwayTeam" not in df.columns and "away_team" in df.columns:
            df["AwayTeam"] = df["away_team"]
        
        # 1. Direct renames
        rename_map = {
            "form_home": "form_home_pts",
            "form_away": "form_away_pts",
            "goals_scored_home": "form_home_gf",
            "goals_scored_away": "form_away_gf",
            "goals_conceded_home": "form_home_ga",
            "goals_conceded_away": "form_away_ga",
            "roster_experience_home": "xi_experience_home",
            "roster_experience_away": "xi_experience_away",
            "advanced_xg_against_home": "advanced_xg_against_home",
            "advanced_xg_against_away": "advanced_xg_against_away",
            "advanced_shots_for_home": "HS",
            "advanced_shots_for_away": "AS",
        }
        
        valid_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=valid_renames)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 2. Derived features
        if "form_home_gf" in df.columns and "form_away_gf" in df.columns:
             df["gf_diff"] = df["form_home_gf"] - df["form_away_gf"]
             
        if "form_home_ga" in df.columns and "form_away_ga" in df.columns:
             df["ga_diff"] = df["form_home_ga"] - df["form_away_ga"]
             
        # 3. Fill specific missing columns expected by models
        # Prompt 2: Split into CRITICAL (no defaults) vs NON-CRITICAL (defaults OK)
        try:
            import json
            from pathlib import Path
            root_dir = Path(__file__).parent.parent.parent
            p = root_dir / "models" / "feature_columns.json"
            
            if p.exists():
                logger.debug(f"Loading feature schema from {p}")
                with open(p) as f:
                    master_cols = json.load(f)
                
                # ── Prompt 2: Separate critical vs non-critical missing ──
                missing_critical_cols = []
                missing_noncritical = {}
                
                for col in master_cols:
                    if col not in df.columns:
                        if col in CRITICAL_FEATURES_1X2:
                            # CRITICAL: do NOT fill with default — mark rows invalid
                            missing_critical_cols.append(col)
                        else:
                            # NON-CRITICAL: apply smart defaults
                            if "synth_xg" in col:
                                derived_xg_col = col.replace("synth_xg", "xg")
                                if derived_xg_col in df.columns:
                                    missing_noncritical[col] = df[derived_xg_col]
                                else:
                                    missing_noncritical[col] = 1.25 if "diff" not in col else 0.0
                            elif "avg_rating" in col:
                                missing_noncritical[col] = 6.5
                            elif "rating_delta" in col:
                                missing_noncritical[col] = 0.0
                            elif "key_players" in col:
                                missing_noncritical[col] = 0
                            elif "xi_experience" in col:
                                missing_noncritical[col] = 90.0
                            elif "ref_goals" in col or "ref_encoded_goals" in col:
                                missing_noncritical[col] = 2.7
                            elif "ref_cards" in col or "ref_encoded_cards" in col:
                                missing_noncritical[col] = 3.5
                            elif "ref_over25" in col:
                                missing_noncritical[col] = 0.55
                            elif "ref_strictness" in col or "ref_experience" in col:
                                missing_noncritical[col] = 0.0
                            elif "formation_score" in col:
                                missing_noncritical[col] = 0.5
                            elif "formation_mismatch" in col or "formation_is_known" in col or "matchup_sample_size" in col:
                                missing_noncritical[col] = 0.0
                            elif "matchup_home_wr" in col:
                                missing_noncritical[col] = 0.44
                            elif "rolling_fouls" in col:
                                missing_noncritical[col] = 12.0
                            elif "rolling_yellows" in col:
                                missing_noncritical[col] = 2.0
                            elif "rolling_corners" in col:
                                missing_noncritical[col] = 5.0
                            elif "rolling_possession" in col:
                                missing_noncritical[col] = 50.0
                            elif "rolling" in col or "prob" in col:
                                missing_noncritical[col] = 0.5
                            elif "ref" in col:
                                missing_noncritical[col] = 0.0
                            elif "streak" in col:
                                missing_noncritical[col] = 0
                            elif col.endswith("H") and "AvgH" in df.columns and any(b in col for b in ["B365", "BW", "IW", "PS", "WH", "VC", "Max"]):
                                missing_noncritical[col] = df["AvgH"]
                            elif col.endswith("D") and "AvgD" in df.columns and any(b in col for b in ["B365", "BW", "IW", "PS", "WH", "VC", "Max"]):
                                missing_noncritical[col] = df["AvgD"]
                            elif col.endswith("A") and "AvgA" in df.columns and any(b in col for b in ["B365", "BW", "IW", "PS", "WH", "VC", "Max"]):
                                missing_noncritical[col] = df["AvgA"]
                            else:
                                missing_noncritical[col] = 0.0
                
                # ── Apply non-critical defaults ──
                if missing_noncritical:
                    new_cols_df = pd.DataFrame(missing_noncritical, index=df.index)
                    df = pd.concat([df, new_cols_df], axis=1)
                
                # ── Handle missing critical columns: add as NaN ──
                for col in missing_critical_cols:
                    df[col] = np.nan
                
                # ── Mark _invalid_for_bet per row ──
                # A row is invalid if ANY critical feature is missing (column absent or NaN)
                if "_invalid_for_bet" not in df.columns:
                    df["_invalid_for_bet"] = False
                
                for col in CRITICAL_FEATURES_1X2:
                    if col in df.columns:
                        df.loc[df[col].isna(), "_invalid_for_bet"] = True
                    else:
                        # Column completely missing → all rows invalid
                        df["_invalid_for_bet"] = True
                
                # ── Capture missing_critical_features reason per row ──
                def _missing_reason(row):
                    missing = []
                    for col in CRITICAL_FEATURES_1X2:
                        if col not in df.columns or pd.isna(row.get(col)):
                            missing.append(col)
                    return ", ".join(missing) if missing else ""
                
                df["_missing_critical_reason"] = df.apply(_missing_reason, axis=1)
                
                # ── Log substitution coverage ──
                n_noncritical_substituted = len(missing_noncritical)
                n_total_noncritical = len([c for c in master_cols if c not in CRITICAL_FEATURES_1X2])
                coverage_pct = round(
                    n_noncritical_substituted / n_total_noncritical * 100, 1
                ) if n_total_noncritical > 0 else 0.0
                
                top_substituted = list(missing_noncritical.keys())[:5]
                logger.info(
                    f"📊 Non-critical substitution coverage: "
                    f"{n_noncritical_substituted}/{n_total_noncritical} cols = {coverage_pct}%. "
                    f"Top substituted: {top_substituted}"
                )
                
                n_invalid = df["_invalid_for_bet"].sum()
                if n_invalid > 0:
                    logger.warning(
                        f"🚫 {n_invalid}/{len(df)} matches marked invalid_for_bet "
                        f"(missing critical: {missing_critical_cols})"
                    )
                
                # Reorder to match master exactly (crucial for LightGBM)
                meta_cols = [c for c in df.columns if c not in master_cols]
                meta_cols = [c for c in meta_cols if c not in master_cols]
                
                df = df[meta_cols + master_cols]
                
        except Exception as e:
            logger.warning(f"Failed to load master feature columns: {e}")
            
        return df

    def _load_history(self) -> pd.DataFrame:
        """
        Load historical data for feature context.
        Prioritizes Parquet for speed and type safety.
        """
        from pathlib import Path
        import pandas as pd
        
        # Try different locations (Parquet first!)
        paths = [
            Path("data/training_data.parquet"),
            Path("data/features_full.parquet"),
            Path("data/training_data.csv"),
            Path("data/historical.csv"),
            Path("data/features_full.csv"),
        ]
        
        for p in paths:
            if p.exists():
                try:
                    logger.info(f"Loading history from {p}...")
                    if p.suffix == ".parquet":
                        df = pd.read_parquet(p)
                    else:
                        # CSV Optimization
                        # Specify types for known columns to save memory
                        dtype_map = {
                            "HomeTeam": "object", "AwayTeam": "object", "League": "object",
                            "FTHG": "float32", "FTAG": "float32",
                            "HST": "float32", "AST": "float32", 
                            "HC": "float32", "AC": "float32",
                            "HY": "float32", "AY": "float32",
                            "HR": "float32", "AR": "float32",
                        }
                        df = pd.read_csv(p, low_memory=False, dtype=dtype_map)
                    
                    # Basic Validation
                    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
                    missing = [c for c in required if c not in df.columns]
                    if missing:
                        logger.warning(f"File {p} missing required columns: {missing}")
                        continue
                        
                    # Parse Date if it's string (avoid warning)
                    if 'Date' in df.columns and df['Date'].dtype == 'object':
                        # Use mixed format to handle potential variations without warning
                        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
                    
                    # Valid load
                    logger.info(f"  -> Loaded {len(df)} rows from {p.name}")
                    return df
                    
                except Exception as e:
                    logger.warning(f"Failed to load {p}: {e}")
                    continue
        
        logger.warning("No historical data found in standard locations.")
        return pd.DataFrame()

    def _df_to_matches(self, df: pd.DataFrame, is_historical: bool = False) -> List['Match']:
        """Convert DataFrame to List[Match]."""
        from stavki.data.schemas import Match, Team, League, MatchStats, MatchLineups, TeamLineup, Player, MatchEnrichment, RefereeInfo
        from datetime import datetime
        import hashlib
        import json
        from stavki.data.processors.normalize import TeamMapper
        
        mapper = TeamMapper.get_instance()
        matches = []
        # Optimization: use itertuples for much faster iteration
        for row in df.itertuples(index=False):
            try:
                # Handle inconsistent column casing via getattr (defaults to None, handled by logic)
                # Helper to check both PascalCase (CSV) and snake_case (Internal)
                p_home = getattr(row, 'HomeTeam', None)
                s_home = getattr(row, 'home_team', None)
                raw_home = str(p_home if p_home is not None else s_home)
                home_name = mapper.map_name(raw_home) or raw_home

                p_away = getattr(row, 'AwayTeam', None)
                s_away = getattr(row, 'away_team', None)
                raw_away = str(p_away if p_away is not None else s_away)
                away_name = mapper.map_name(raw_away) or raw_away

                home = Team(name=home_name)
                away = Team(name=away_name)
                
                # Date/Time
                if is_historical:
                    date_val = getattr(row, 'Date', None)
                    if date_val is None:
                         date_val = getattr(row, 'date', None)
                         
                    if hasattr(date_val, 'date'):
                        date_str = date_val.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date_val).split()[0]
                    commence = datetime.strptime(date_str, "%Y-%m-%d")

                    
                    # Generate ID for historical
                    key = f"{home.normalized_name}_{away.normalized_name}_{date_str}"
                    mid = hashlib.md5(key.encode()).hexdigest()[:12]
                    
                    # Scores
                    val_hg = getattr(row, 'FTHG', None)
                    val_ag = getattr(row, 'FTAG', None)
                    home_score = int(val_hg) if pd.notna(val_hg) else None
                    away_score = int(val_ag) if pd.notna(val_ag) else None
                    
                else:
                    # Upcoming
                    commence = getattr(row, 'commence_time', None)
                    if isinstance(commence, str):
                        commence = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    
                    # Use existing ID
                    mid = str(getattr(row, 'event_id', getattr(row, 'match_id', 'unknown')))
                    home_score = None
                    away_score = None
                
                # League
                league_str = str(getattr(row, 'League', getattr(row, 'league', 'unknown'))).lower()
                
                # Robust mapping
                league_map = {
                    "epl": League.EPL,
                    "premier league": League.EPL,
                    "premier_league": League.EPL,
                    "soccer_epl": League.EPL,
                    "laliga": League.LA_LIGA,
                    "la liga": League.LA_LIGA,
                    "soccer_spain_la_liga": League.LA_LIGA,
                    "bundesliga": League.BUNDESLIGA,
                    "soccer_germany_bundesliga": League.BUNDESLIGA,
                    "seriea": League.SERIE_A,
                    "serie a": League.SERIE_A,
                    "soccer_italy_serie_a": League.SERIE_A,
                    "ligue1": League.LIGUE_1,
                    "ligue 1": League.LIGUE_1,
                    "soccer_france_ligue_one": League.LIGUE_1,
                    "championship": League.CHAMPIONSHIP,
                    "soccer_efl_champ": League.CHAMPIONSHIP,
                }
                
                league_enum = league_map.get(league_str, League.EPL) # Default to EPL if unknown
                
                # Stats (Historical Only)
                stats = None
                if is_historical:
                     v_hs = getattr(row, 'HS', None)
                     v_as = getattr(row, 'AS', None)
                     if pd.notna(v_hs) and pd.notna(v_as):
                         try:
                             stats = MatchStats(
                                 match_id=mid,
                                 shots_home=int(v_hs) if pd.notna(v_hs) else None,
                                 shots_away=int(v_as) if pd.notna(v_as) else None,
                                 shots_on_target_home=int(getattr(row, 'HST')) if pd.notna(getattr(row, 'HST', None)) else None,
                                 shots_on_target_away=int(getattr(row, 'AST')) if pd.notna(getattr(row, 'AST', None)) else None,
                                 corners_home=int(getattr(row, 'HC')) if pd.notna(getattr(row, 'HC', None)) else None,
                                 corners_away=int(getattr(row, 'AC')) if pd.notna(getattr(row, 'AC', None)) else None,
                                 fouls_home=int(getattr(row, 'HF')) if pd.notna(getattr(row, 'HF', None)) else None,
                                 fouls_away=int(getattr(row, 'AF')) if pd.notna(getattr(row, 'AF', None)) else None,
                                 yellow_cards_home=int(getattr(row, 'HY')) if pd.notna(getattr(row, 'HY', None)) else None,
                                 yellow_cards_away=int(getattr(row, 'AY')) if pd.notna(getattr(row, 'AY', None)) else None,
                                 red_cards_home=int(getattr(row, 'HR')) if pd.notna(getattr(row, 'HR', None)) else None,
                                 red_cards_away=int(getattr(row, 'AR')) if pd.notna(getattr(row, 'AR', None)) else None,
                             )
                             
                             # Extract xG if available directly from CSV
                             if hasattr(row, 'xg_home') and pd.notna(row.xg_home):
                                 stats.xg_home = float(row.xg_home)
                             if hasattr(row, 'xg_away') and pd.notna(row.xg_away):
                                 stats.xg_away = float(row.xg_away)
                         except Exception:
                             pass # Robustness
                             
                # Lineups and Enrichment (Historical Phase 2)
                lineups = None
                enrichment = None
                if is_historical:
                    # Referee
                    ref = getattr(row, 'referee', None) or getattr(row, 'Referee', None)
                    if pd.notna(ref) and ref:
                        enrichment = MatchEnrichment(match_id=mid, referee=RefereeInfo(name=str(ref)))
                        
                    # Lineups
                    lh, la = getattr(row, 'lineup_home', None), getattr(row, 'lineup_away', None)
                    if pd.notna(lh) and pd.notna(la) and lh and la:
                        try:
                            def parse_team_lineup(team_name, formation, lineup_str):
                                players = []
                                if isinstance(lineup_str, str):
                                    js = json.loads(lineup_str)
                                    for p in js:
                                        # Support varied JSON formats (SportMonks vs historical dict)
                                        pid = str(p.get("player_id", p.get("id", "0")))
                                        pname = p.get("player_name", p.get("name", "Unknown"))
                                        prat = float(p.get("rating", 0.0)) if p.get("rating") else None
                                        ppos = p.get("position", None)
                                        players.append(Player(id=pid, name=pname, position=ppos, rating=prat))
                                return TeamLineup(team_name=team_name, formation=str(formation) if pd.notna(formation) else None, starting_xi=players)
                            
                            lineups = MatchLineups(
                                match_id=mid,
                                home=parse_team_lineup(home_name, getattr(row, 'formation_home', None), lh),
                                away=parse_team_lineup(away_name, getattr(row, 'formation_away', None), la)
                            )
                        except Exception as e:
                            pass # Skip corrupt JSONs

                                
                m = Match(
                    id=mid,
                    home_team=home,
                    away_team=away,
                    league=league_enum,
                    commence_time=commence,
                    home_score=home_score,
                    away_score=away_score,
                    enrichment=enrichment,
                    lineups=lineups,
                    stats=stats,
                    source="historical" if is_historical else "api"
                )
                matches.append(m)
                
            except Exception as e:
                # logger.warning(f"Failed to convert match row: {e}")
                continue
                
        return matches
    
    def _load_calibrator(self):
        """Load fitted calibrator from models directory if available."""
        try:
            import joblib
            calibrator_path = Path(__file__).resolve().parent.parent.parent / "models" / "calibrator.joblib"
            if calibrator_path.exists():
                self._calibrator = joblib.load(calibrator_path)
                if hasattr(self._calibrator, 'is_fitted') and self._calibrator.is_fitted:
                    n = len(getattr(self._calibrator, 'calibrators', {}))
                    logger.info(f"  Loaded calibrator ({n} outcome calibrators)")
                else:
                    logger.info("  Calibrator found but not fitted, skipping")
                    self._calibrator = None
            else:
                logger.debug("  No calibrator.joblib found, using raw probabilities")
        except Exception as e:
            logger.warning(f"  Failed to load calibrator: {e}")
            self._calibrator = None
    
    def _get_predictions(
        self,
        matches_df: pd.DataFrame,
        features_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Get ensemble predictions for all matches.
        
        Returns:
            Nested dict: {match_id: {market: {outcome: prob}}}
            Example: {"19427718": {"1x2": {"home": 0.3, "draw": 0.3, "away": 0.4}, ...}}
        """
        predictions: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        try:
            import sys, os
            root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if root_path not in sys.path:
                sys.path.insert(0, root_path)

            from stavki.models.ensemble.predictor import EnsemblePredictor
            from stavki.models.base import Market, Prediction, MatchPredictions
            
            # Lazy load ensemble and cache it
            if not getattr(self, "ensemble", None):
                self.ensemble = self._load_ensemble()
            
            ensemble = self.ensemble
            
            if ensemble and ensemble.models:
                logger.info(f"  Using ensemble with {len(ensemble.models)} models")
                
                # Compatibility: Add legacy column aliases if missing
                # CRITICAL FIX for High EV: Normalize team names via TeamMapper
                from stavki.data.processors.normalize import TeamMapper
                _mapper = TeamMapper.get_instance()
                
                if "home_team" in features_df.columns:
                    features_df["home_team"] = features_df["home_team"].apply(lambda n: _mapper.map_name(n) or n)
                    features_df["HomeTeam"] = features_df["home_team"]
                
                if "away_team" in features_df.columns:
                    features_df["away_team"] = features_df["away_team"].apply(lambda n: _mapper.map_name(n) or n)
                    features_df["AwayTeam"] = features_df["away_team"]

                if "League" not in features_df.columns and "league" in features_df.columns:
                    features_df["League"] = features_df["league"]
                
                # Build match_id → event_id reverse lookup
                # Models generate match_id from team names + date (a hash string),
                # but the pipeline uses event_id (SportMonks fixture_id integer).
                # We need to remap predictions back to event_id for the join in _find_value_bets.
                from stavki.utils import generate_match_id
                mid_to_eid = {}
                eid_col = "event_id" if "event_id" in features_df.columns else None
                ht_col = "HomeTeam" if "HomeTeam" in features_df.columns else "home_team"
                at_col = "AwayTeam" if "AwayTeam" in features_df.columns else "away_team"
                dt_col = "Date" if "Date" in features_df.columns else "commence_time"
                
                if eid_col:
                    # Vectorized fillna for xG columns
                    xg_cols = [c for c in ["xg_home", "xg_away", "xg_diff"] if c in features_df.columns]
                    if xg_cols:
                        features_df[xg_cols] = features_df[xg_cols].fillna(0.0)

                    # List comprehension for fast ID generation (avoid iterrows overhead)
                    # Extract columns to lists for faster iteration
                    h_list = features_df[ht_col].astype(str).tolist()
                    a_list = features_df[at_col].astype(str).tolist()
                    d_list = features_df[dt_col].tolist()
                    e_list = features_df[eid_col].astype(str).tolist()
                    
                    mid_to_eid = {
                        generate_match_id(h, a, d): e
                        for h, a, d, e in zip(h_list, a_list, d_list, e_list)
                    }
                    logger.info(f"  Built match_id→event_id mapping for {len(mid_to_eid)} matches")
                
                preds = ensemble.predict(features_df)
                
                # Apply calibration if calibrator is available
                if self._calibrator is not None:
                    preds = self._calibrator.calibrate(preds)
                    logger.info("  Applied probability calibration")
                
                for pred in preds:
                    # Remap model's match_id to pipeline's event_id
                    event_id = mid_to_eid.get(pred.match_id, pred.match_id)
                    market_key = pred.market.value if hasattr(pred.market, 'value') else str(pred.market)
                    
                    if event_id not in predictions:
                        predictions[event_id] = {}
                    if market_key not in predictions[event_id]:
                        predictions[event_id][market_key] = {}
                    
                    # 🛡️ Epistemic Uncertainty Subtraction (Global Telegram Feed)
                    # Deduct PyTorch MCMC standard deviations directly from output probability cache
                    metadata = getattr(pred, "metadata", {}) or {}
                    if metadata and "home_std" in metadata:
                        std_home = metadata.get("home_std", 0.0)
                        std_draw = metadata.get("draw_std", 0.0)
                        std_away = metadata.get("away_std", 0.0)
                        
                        probs = pred.probabilities.copy()
                        if "home" in probs: probs["home"] = max(0.01, probs["home"] - std_home)
                        if "draw" in probs: probs["draw"] = max(0.01, probs["draw"] - std_draw)
                        if "away" in probs: probs["away"] = max(0.01, probs["away"] - std_away)
                        
                        # Re-normalize
                        total = sum(probs.values())
                        if total > 0:
                            for k in probs:
                                probs[k] = probs[k] / total
                                
                        predictions[event_id][market_key].update(probs)
                    else:
                        predictions[event_id][market_key].update(pred.probabilities)
                
                if predictions:
                    n_markets = sum(len(mkts) for mkts in predictions.values())
                    logger.info(f"  → {len(predictions)} matches predicted ({n_markets} market slots)")
                    return predictions
            
        except Exception as e:
            logger.warning(f"Ensemble prediction failed: {e}")
            import traceback
            traceback.print_exc()
        
        if not predictions:
            logger.error(
                "No predictions available — models not trained or not loadable. "
                "Run the training pipeline first: stavki train"
            )
        
        return predictions
    
    def _load_ensemble(self):
        """Load ensemble predictor with saved models and weights."""
        from stavki.models.ensemble.predictor import EnsemblePredictor
        
        models_dir = Path("models")
        if not models_dir.exists():
            logger.warning(f"Models directory not found: {models_dir}")
            return None
        
        ensemble = EnsemblePredictor()
        
        # 1. Load configuration from league_weights.json (preferred) or league_config.json (legacy)
        config_path = models_dir / "league_weights.json"
        if not config_path.exists():
            config_path = models_dir / "league_config.json"
            
        if config_path.exists():
            try:
                ensemble.load_weights(config_path)
            except Exception as e:
                logger.warning(f"  Failed to load {config_path.name}: {e}")
        else:
            logger.warning(f"  Config not found at {config_path}, using defaults.")
        
        # Try to load saved model files
        # Include _config.json for split-file models (NeuralMultiTask)
        model_files = list(models_dir.glob("*.pkl")) + \
                      list(models_dir.glob("*.joblib")) + \
                      list(models_dir.glob("*_config.json"))

        logger.info(f"  Found {len(model_files)} model files to load: {[f.name for f in model_files]}")
        
        for model_path in model_files:
            # Skip non-model files
            if "calibrator" in model_path.name or "preproc" in model_path.name:
                continue
                
            try:
                from stavki.models.base import BaseModel
                
                # Handle split-file config paths
                if model_path.name.endswith("_config.json"):
                    # Skip known non-model configs
                    if "league" in model_path.name or "training" in model_path.name:
                        continue
                        
                    # Strip _config.json to get base stem (e.g. NeuralMultiTask)
                    # We pass the abstract base path which load() uses to find config
                    base_name = model_path.name.replace("_config.json", "")
                    load_path = model_path.parent / base_name
                    model = BaseModel.load(load_path)
                else:
                    model = BaseModel.load(model_path)

                if model:
                    ensemble.add_model(model)
                    logger.info(f"  Loaded model: {model.name} from {model_path.name}")
                else:
                    logger.warning(f"  Loaded {model_path.name} but result was None/Empty")
            except Exception as e:
                logger.warning(f"  Could not load {model_path.name}: {e}")
                import traceback
                traceback.print_exc()
                
        # --- Load Shadow Models (V3 Watcher) ---
        try:
            from stavki.models.v3_transformer_wrapper import V3WatcherWrapper
            # Initialize V3 Neural Transformer Model
            v3_path = models_dir / "deep_interaction_v3.pth"
            # Wrapper handles missing file gracefully
            v3_watcher = V3WatcherWrapper(model_path=v3_path)
            ensemble.add_shadow_model(v3_watcher)
        except Exception as e:
            logger.warning(f"Failed to initialize V3 Watcher: {e}")
        
        return ensemble if ensemble.models else None
    
    def _select_best_prices(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Select best price per outcome from all bookmakers."""
        from stavki.strategy import check_outlier_odds
        
        if "outcome_price" not in odds_df.columns:
            # Check for wide format (SportMonks style columns)
            if "home_odds" in odds_df.columns:
                # Melt into long format
                id_vars = [c for c in odds_df.columns if c not in ["home_odds", "draw_odds", "away_odds", "_enrichment"]]
                
                melted = odds_df.melt(
                    id_vars=id_vars,
                    value_vars=["home_odds", "draw_odds", "away_odds"],
                    var_name="outcome_temp",
                    value_name="outcome_price"
                )
                
                # Map temp names to localized Outcome names (must match model output keys: home, draw, away)
                outcome_map = {
                    "home_odds": "home",
                    "draw_odds": "draw", 
                    "away_odds": "away"
                }
                melted["outcome_name"] = melted["outcome_temp"].map(outcome_map)
                melted = melted.drop(columns=["outcome_temp"])
                
                # Ensure outcome_price is numeric
                melted["outcome_price"] = pd.to_numeric(melted["outcome_price"], errors="coerce")
                
                # Map bookmaker (approximate using home_bookmaker)
                if "home_bookmaker" in melted.columns:
                    melted["bookmaker_title"] = melted["home_bookmaker"]
                
                odds_df = melted
                
            # Try to find odds column (Legacy fallback)
            elif any("odds" in c.lower() or "price" in c.lower() for c in odds_df.columns):
                odds_cols = [c for c in odds_df.columns if "odds" in c.lower() or "price" in c.lower()]
                if odds_cols:
                    odds_df = odds_df.rename(columns={odds_cols[0]: "outcome_price"})
            else:
                logger.warning("No odds column found")
                return odds_df
        
        # Group by event + outcome, get max
        group_cols = ["event_id"]
        if "outcome_name" in odds_df.columns:
            group_cols.append("outcome_name")
        if "market_key" in odds_df.columns:
            group_cols.append("market_key")
        
        if len(group_cols) > 1:
            idx = odds_df.groupby(group_cols)["outcome_price"].idxmax()
            best = odds_df.loc[idx].reset_index(drop=True)
        else:
            best = odds_df.copy()
        
        return best
    
    def _compute_market_probs(
        self,
        best_prices: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """Compute no-vig probabilities from best prices using strict vectorization."""
        market_probs = {}
        
        if "event_id" not in best_prices.columns or best_prices.empty:
            return market_probs
            
        prices = best_prices.copy()
        # Fast vectorized implied probability
        prices['implied_prob'] = np.where(prices['outcome_price'] > 0, 1.0 / prices['outcome_price'], 0.0)
        
        # Calculate market margin/total per event safely
        totals = prices.groupby('event_id')['implied_prob'].sum().reset_index()
        totals.rename(columns={'implied_prob': 'total_prob'}, inplace=True)
        
        # Broadcast margin and normalize
        prices = prices.merge(totals, on='event_id')
        prices['no_vig_prob'] = np.where(prices['total_prob'] > 0, prices['implied_prob'] / prices['total_prob'], 0.0)
        
        # Convert to dictionary cleanly mapping event_id -> {outcome: prob}
        for event_id, group in prices.groupby('event_id'):
            market_probs[str(event_id)] = dict(zip(group['outcome_name'], group['no_vig_prob']))
            
        return market_probs
    
    # Map outcome names to their Market enum .value for structured lookups.
    # Must align with Market.MATCH_WINNER.value="1x2", Market.OVER_UNDER.value="over_under", etc.
    # Map outcome names to their Market enum .value for structured lookups.
    # Must align with Market.MATCH_WINNER.value="1x2", Market.OVER_UNDER.value="over_under", etc.
    OUTCOME_TO_MARKET = {
        "home": "1x2", "draw": "1x2", "away": "1x2",
        "over_2.5": "over_under", "under_2.5": "over_under",
        "yes": "btts", "no": "btts",
        # Combo Markets
        "home & yes": "result_btts", "draw & yes": "result_btts", "away & yes": "result_btts",
        "home & no": "result_btts", "draw & no": "result_btts", "away & no": "result_btts",
        # Double Chance
        "1x": "double_chance", "12": "double_chance", "x2": "double_chance",
        "home/draw": "double_chance", "home/away": "double_chance", "draw/away": "double_chance",
    }

    def _find_value_bets(
        self,
        matches_df: pd.DataFrame,
        model_probs: Dict[str, Dict[str, Dict[str, float]]],
        market_probs: Dict[str, Dict[str, float]],
        best_prices: pd.DataFrame,
    ) -> List[BetCandidate]:
        """Find all value betting opportunities.
        
        Args:
            model_probs: {match_id: {market: {outcome: prob}}}
            market_probs: {match_id: {outcome: no_vig_prob}}
            best_prices: DataFrame with event_id, outcome_name, outcome_price
        """
        from stavki.strategy import check_model_market_divergence, calculate_justified_score
        
        candidates = []
        
        # O(N) HashMap caching for prices (avoid O(N*M) nested DataFrame scans)
        prices_map = {str(k): v.to_dict('records') for k, v in best_prices.groupby(best_prices["event_id"].astype(str))}
        
        # Generator optimization: Replace memory-heavy .to_dict() with memory-efficient iteration
        for match_idx in range(len(matches_df)):
            match = matches_df.iloc[match_idx]
            event_id = str(match.get("event_id"))
            home = match.get("home_team", "Home")
            away = match.get("away_team", "Away")
            league = match.get("league", match.get("sport_key", "unknown"))
            
            if event_id not in model_probs:
                continue
            
            # ── Prompt 2: Skip matches marked invalid_for_bet ──
            if "_invalid_for_bet" in matches_df.columns:
                invalid_val = match.get("_invalid_for_bet", False)
                if invalid_val is True or invalid_val == 1:
                    continue
            
            event_markets = model_probs[event_id]  # {market: {outcome: prob}}
            event_market_probs = market_probs.get(event_id, {})
            
            event_prices = prices_map.get(event_id, [])
            
            for price_row in event_prices:
                outcome = price_row.get("outcome_name")
                odds = price_row.get("outcome_price", 2.0)
                bookmaker = price_row.get(
                    "bookmaker_title",
                    price_row.get("bookmaker_key", "Unknown"),
                )
                
                if outcome is None or odds <= 1.0:
                    continue
                
                # Resolve which market this outcome belongs to
                market_key = self.OUTCOME_TO_MARKET.get(
                    outcome.lower(), price_row.get("market_key")
                )
                if not market_key:
                    continue
                
                # Look up model probability from the correct market
                market_probs_dict = event_markets.get(market_key, {})
                p_model = market_probs_dict.get(outcome)
                
                # Robust key matching (handle case differences)
                if p_model is None:
                    p_model = (
                        market_probs_dict.get(outcome.lower())
                        or market_probs_dict.get(outcome.title())
                    )
                    
                # DERIVED PROBABILITIES for Combo Markets (Bob's Request)
                # If we don't have authorized model output for "Result & BTTS", we approximate it
                # to ensure we don't miss high-value opportunities provided by the data collector.
                if p_model is None and market_key == "result_btts":
                    # Parse "Home & Yes"
                    normalized_outcome = outcome.lower()
                    
                    # Get base probabilities
                    probs_1x2 = event_markets.get("1x2", {})
                    probs_btts = event_markets.get("btts", {})
                    
                    p_res = None
                    p_btts = None
                    
                    if "home" in normalized_outcome or "1" in normalized_outcome:
                        p_res = probs_1x2.get("home")
                    elif "draw" in normalized_outcome or "x" in normalized_outcome:
                        p_res = probs_1x2.get("draw")
                    elif "away" in normalized_outcome or "2" in normalized_outcome:
                        p_res = probs_1x2.get("away")
                        
                    if "yes" in normalized_outcome:
                        p_btts = probs_btts.get("yes")
                    elif "no" in normalized_outcome:
                        p_btts = probs_btts.get("no")
                        
                    if p_res and p_btts:
                        # Naive independence assumption: P(A & B) = P(A) * P(B)
                        # We apply a penalty discount for correlation risk
                        # e.g. Favorites winning + BTTS No is correlated.
                        # For now, just raw multiplication to surface the bet.
                        p_model = p_res * p_btts
                        
                        # Add to the dict so we don't recompute
                        if "result_btts" not in event_markets:
                            event_markets["result_btts"] = {}
                        event_markets["result_btts"][outcome] = p_model
                
                if p_model is None and market_key == "double_chance":
                    probs_1x2 = event_markets.get("1x2", {})
                    normalized_outcome = outcome.lower()
                    if "1x" in normalized_outcome or "home/draw" in normalized_outcome:
                        p_model = (probs_1x2.get("home", 0) + probs_1x2.get("draw", 0))
                    elif "x2" in normalized_outcome or "draw/away" in normalized_outcome:
                        p_model = (probs_1x2.get("away", 0) + probs_1x2.get("draw", 0))
                    elif "12" in normalized_outcome or "home/away" in normalized_outcome:
                        p_model = (probs_1x2.get("home", 0) + probs_1x2.get("away", 0))

                if p_model is None:
                    continue
                

                p_market = event_market_probs.get(outcome, 1.0 / odds)
                
                # Blend model and market probabilities
                p_blended = self._blender.blend(p_model, p_market, league)
                
                # EV = p * odds - 1
                ev = p_blended * odds - 1.0
                edge = p_blended - (1.0 / odds)
                
                if ev < self.config.min_ev:
                    continue
                    
                # Python Bug Buster / Strategy Fix: Hard cap block
                # Even for Tier-3 leagues, a 35% edge indicates missing properties, injured squads, or ghost match bugs.
                # Protect bankroll against logic bugs.
                if ev > 0.35:
                    logger.warning(
                        f"🚨 BLOCKED: Mathematically improbable EV ({ev:.2%}) for {event_id} ({outcome}). "
                        f"Odds: {odds}, Prob: {p_model:.2%} | Market: {p_market:.2%}. Skipping to preserve bankroll safety."
                    )
                    continue
                
                # Quality checks
                _, divergence, div_level = check_model_market_divergence(
                    p_model, p_market,
                )
                justified = calculate_justified_score(p_model, p_market, odds, ev)
                
                candidates.append(BetCandidate(
                    match_id=event_id,
                    home_team=home,
                    away_team=away,
                    league=league,
                    kickoff=None,
                    market=market_key,
                    selection=outcome,
                    odds=odds,
                    bookmaker=bookmaker,
                    model_prob=p_model,
                    market_prob=p_market,
                    blended_prob=p_blended,
                    ev=ev,
                    edge=edge,
                    stake_pct=0.0,
                    stake_amount=0.0,
                    confidence=abs(p_model - p_market),
                    justified_score=justified,
                    divergence_level=div_level,
                ))
        
        candidates.sort(key=lambda x: -x.ev)
        return candidates
    
    def _apply_filters(self, candidates: List[BetCandidate]) -> List[BetCandidate]:
        """Apply betting filters."""
        filtered = []
        
        for bet in candidates:
            # Divergence filter: only block extreme divergence when justified score is low
            # High-justified contrarian bets may still have genuine edge
            if bet.divergence_level == "extreme" and bet.justified_score < 50:
                continue
            
            # Justified score filter
            if bet.justified_score < 30:
                continue
            
            # Confidence filter
            if bet.confidence < self.config.min_confidence:
                continue
            
            filtered.append(bet)
        
        return filtered
    
    def _calculate_stakes(self, bets: List[BetCandidate]) -> List[BetCandidate]:
        """Calculate stake amounts for all bets."""
        from stavki.strategy import EVResult
        
        for bet in bets:
            ev_result = EVResult(
                match_id=bet.match_id,
                market=bet.market,
                selection=bet.selection,
                model_prob=bet.blended_prob,
                odds=bet.odds,
                ev=bet.ev,
                edge_pct=bet.edge,
                implied_prob=1.0 / bet.odds,
            )
            
            stake_result = self._staker.calculate_stake(
                ev_result,
                league=bet.league,
                apply_limits=True
            )
            
            bet.stake_pct = stake_result.stake_pct
            bet.stake_amount = stake_result.stake_amount
            
            if bet.stake_amount == 0:
                logger.debug(f"Matches {bet.match_id}: Stake $0. Reason: {stake_result.reason} (EV={bet.ev:.1%}, Odds={bet.odds})")
        
        # Filter out zero stakes
        # NO! Don't filter them yet, let the bot decide or log them. 
        # Actually, for the pipeline return, we usually want actionable bets.
        # But for debugging "why 0", we might want to see them if we change logic.
        # For now, keep filtering but we logged the reason above.
        return [b for b in bets if b.stake_amount > 0]
    
    def _save_output(self, bets: List[BetCandidate]):
        """Save pipeline output to file."""
        output_dir = self.config.output_dir / "bets"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"bets_{timestamp}.json"
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "bankroll": self.bankroll,
            "total_bets": len(bets),
            "total_stake": sum(b.stake_amount for b in bets),
            "bets": [b.to_dict() for b in bets],
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        
        # Also save as optimized CSV for users
        csv_filepath = output_dir / f"bets_{timestamp}.csv"
        
        # Sort by EV descending
        sorted_bets = sorted(bets, key=lambda x: x.ev, reverse=True)
        
        csv_rows = []
        for b in sorted_bets:
            csv_rows.append({
                "Match": f"{b.home_team} vs {b.away_team}",
                "Time": b.kickoff.strftime("%Y-%m-%d %H:%M") if b.kickoff else "TBD",
                "League": str(b.league).split(".")[-1] if hasattr(b.league, "name") else str(b.league),
                "Selection": b.selection,
                "Odds": round(b.odds, 2),
                "Bookmaker": b.bookmaker,
                "EV (%)": round(b.ev * 100, 1),
                "Stake ($)": round(b.stake_amount, 2),
                "Confidence": round(b.confidence, 2),
            })
            
        pd.DataFrame(csv_rows).to_csv(csv_filepath, index=False)
        logger.info(f"Saved {len(bets)} bets to {filepath} and {csv_filepath}")


def run_daily_pipeline(
    leagues: List[str] = None,
    bankroll: float = 1000.0,
    min_ev: float = 0.03,
) -> List[BetCandidate]:
    """Convenience function to run daily pipeline."""
    config = PipelineConfig(
        leagues=leagues or ["soccer_epl"],
        min_ev=min_ev,
    )
    pipeline = DailyPipeline(config=config, bankroll=bankroll)
    return pipeline.run()
