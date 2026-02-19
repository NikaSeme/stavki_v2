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


@dataclass
class PipelineConfig:
    """Configuration for daily pipeline."""
    # Data
    leagues: List[str] = field(default_factory=lambda: [
        "soccer_epl", 
        "soccer_spain_la_liga", 
        "soccer_germany_bundesliga", 
        "soccer_italy_serie_a", 
        "soccer_france_ligue_one"
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
                        logger.error(f"SportMonks fixture fetch failed: {e}")
                
                if not matches and oa_collector:
                    logger.info("Falling back to OddsAPI for fixtures...")
                    matches = oa_collector.fetch_matches(league, include_odds=True, max_hours_ahead=self.config.scan_window_hours)
                
                if not matches:
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
                        logger.info(f"Fetched SportMonks odds for {len(sm_odds)} matches")
                    except Exception as e:
                        logger.error(f"SportMonks odds fetch failed: {e}")

                # C. Build Rows
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
                                # Only set if not already set (though SM is iterated first here)
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
                        # Prefer Betfair for 1X2 if available? 
                        # Original logic: bf_odds was checked first.
                        # Let's override 1X2 if we have Betfair (it's often sharper)
                        best_snap = bf_odds[m.id][0]
                        row.update({
                            "home_odds": best_snap.home_odds,
                            "draw_odds": best_snap.draw_odds,
                            "away_odds": best_snap.away_odds,
                            "home_bookmaker": best_snap.bookmaker,
                            "away_bookmaker": best_snap.bookmaker,
                        })
                        primary_found = True

                    # 3. OddsAPI Fallback
                    if not primary_found and m.source == "odds_api" and oa_collector:
                         pass # Logic as before (placeholder)

                    if primary_found:
                        rows.append(row)
            
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
        
        def _process_single_match(row):
            """Helper to process a single match in a thread."""
            idx = row.name
            match_id = row.get("event_id", f"{row.get('home_team', '')}_{row.get('away_team', '')}")
            fixture_id = row.get("fixture_id") or row.get("source_id")
            
            if not fixture_id:
                return None
            
            try:
                enrichment = MatchEnrichment()
                
                # Weather
                try:
                    weather = client.get_fixture_weather(int(fixture_id))
                    if weather:
                        enrichment.weather = WeatherInfo(
                            temperature_c=weather.get("temperature"),
                            wind_speed_ms=weather.get("wind"),
                            humidity_pct=weather.get("humidity"),
                            precipitation_mm=weather.get("precipitation"),
                            description=weather.get("description"),
                        )
                except Exception as e:
                    # Log at debug to avoid spamming console
                    logger.debug(f"  Weather fetch failed for {fixture_id}: {e}")
                
                # SM Odds (Multi-Market)
                try:
                    odds = client.get_fixture_odds(int(fixture_id), market="ALL")
                    if odds:
                        for o in odds:
                            market_type = o.get("market_type", "1x2")
                            vals = o.get("odds", {})
                            
                            if market_type == "1x2":
                                if vals.get("home") and vals.get("draw") and vals.get("away"):
                                    enrichment.sm_odds_home = float(vals["home"])
                                    enrichment.sm_odds_draw = float(vals["draw"])
                                    enrichment.sm_odds_away = float(vals["away"])
                            
                            elif market_type == "corners_1x2":
                                if vals.get("home") and vals.get("draw") and vals.get("away"):
                                    enrichment.sm_corners_home = float(vals["home"])
                                    enrichment.sm_corners_draw = float(vals["draw"])
                                    enrichment.sm_corners_away = float(vals["away"])
                                    
                            elif market_type == "btts":
                                if vals.get("yes") and vals.get("no"):
                                    enrichment.sm_btts_yes = float(vals["yes"])
                                    enrichment.sm_btts_no = float(vals["no"])

                except Exception as e:
                    logger.debug(f"  SM odds fetch failed for {fixture_id}: {e}")
                
                return match_id, enrichment
                
            except Exception as e:
                logger.debug(f"Enrichment failed for match {match_id}: {e}")
                return None

        # Execute in parallel
        # 10 workers is usually safe for API calls (mostly I/O)
        # SportMonks rate limit will be defining factor
        max_workers = min(10, len(matches_df)) if len(matches_df) > 0 else 1
        logger.info(f"Enriching {len(matches_df)} matches using {max_workers} threads...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single_match, row) for _, row in matches_df.iterrows()]
            
            processed_count = 0
            for future in as_completed(futures):
                result = future.result()
                processed_count += 1
                if result:
                    m_id, enrich = result
                    enrichments[m_id] = enrich
                
                if processed_count % 5 == 0:
                    logger.info(f"  → Enriched {processed_count}/{len(matches_df)} matches...")
        
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
        """
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
            "advanced_xg_for_home": "synth_xg_home",
            "advanced_xg_for_away": "synth_xg_away",
            "advanced_xg_against_home": "advanced_xg_against_home", # Explicit pass-through or rename if needed
            "advanced_xg_against_away": "advanced_xg_against_away",
            "advanced_shots_for_home": "HS", # Approximation if HS missing
            "advanced_shots_for_away": "AS",
        }
        
        # Apply renames for columns that exist
        valid_renames = {k: v for k, v in rename_map.items() if k in df.columns}
        df = df.rename(columns=valid_renames)
        
        # Remove duplicates if rename caused collisions (e.g. synth_xg_home existed + was renamed to)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 2. Derived features
        if "form_home_gf" in df.columns and "form_away_gf" in df.columns:
             df["gf_diff"] = df["form_home_gf"] - df["form_away_gf"]
             
        if "form_home_ga" in df.columns and "form_away_ga" in df.columns:
             df["ga_diff"] = df["form_home_ga"] - df["form_away_ga"]
             
        # 3. Fill specific missing columns expected by models (defaults)
        # Load the master feature list from JSON if possible, else use hardcoded defaults
        # Load the master feature list from JSON if possible, else use hardcoded defaults
        try:
            # json and Path already imported or available
            import json
            from pathlib import Path
            # Resolve path relative to this file
            root_dir = Path(__file__).parent.parent.parent
            p = root_dir / "models" / "feature_columns.json"
            
            if p.exists():
                logger.debug(f"Loading feature schema from {p}")
                with open(p) as f:
                    master_cols = json.load(f)
                
                # Identify missing columns
                missing_cols = {}
                for col in master_cols:
                    if col not in df.columns:
                        # Smart defaults for features not populated by _build_features
                        if "rolling" in col or "prob" in col:
                            missing_cols[col] = 0.5  # Neutral for rolling/probability features
                        elif "ref" in col:
                            missing_cols[col] = 0.0
                        elif "streak" in col:
                            missing_cols[col] = 0
                        else:
                            missing_cols[col] = 0.0
                
                # Add all missing columns at once to avoid fragmentation
                if missing_cols:
                    new_cols_df = pd.DataFrame(missing_cols, index=df.index)
                    df = pd.concat([df, new_cols_df], axis=1)
                
                # Reorder to match master exactly (crucial for LightGBM)
                # Keep match_id/event_id for merging
                meta_cols = [c for c in df.columns if c not in master_cols]
                # Avoid duplicates in meta_cols if they are in master_cols (shouldn't happen but safe)
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
        from stavki.data.schemas import Match, Team, League, MatchStats
        from datetime import datetime
        import hashlib
        
        matches = []
        for _, row in df.iterrows():
            try:
                # Teams
                home = Team(name=str(row.get('HomeTeam', row.get('home_team'))))
                away = Team(name=str(row.get('AwayTeam', row.get('away_team'))))
                
                # Date/Time
                if is_historical:
                    date_val = row.get('Date')
                    if hasattr(date_val, 'date'):
                        date_str = date_val.strftime("%Y-%m-%d")
                    else:
                        date_str = str(date_val).split()[0]
                    commence = datetime.strptime(date_str, "%Y-%m-%d")
                    
                    # Generate ID for historical
                    key = f"{home.normalized_name}_{away.normalized_name}_{date_str}"
                    mid = hashlib.md5(key.encode()).hexdigest()[:12]
                    
                    # Scores
                    home_score = int(row['FTHG']) if pd.notna(row.get('FTHG')) else None
                    away_score = int(row['FTAG']) if pd.notna(row.get('FTAG')) else None
                    
                else:
                    # Upcoming
                    commence = row.get('commence_time')
                    if isinstance(commence, str):
                        commence = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                    
                    # Use existing ID
                    mid = str(row.get('event_id', row.get('match_id', 'unknown')))
                    home_score = None
                    away_score = None
                
                # League
                league_str = str(row.get('League', row.get('league', 'unknown'))).lower()
                
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
                     # Check if stats columns exist (HS = Home Shots, AS = Away Shots)
                     if pd.notna(row.get('HS')) and pd.notna(row.get('AS')):
                         try:
                             stats = MatchStats(
                                 match_id=mid,
                                 shots_home=int(row['HS']) if pd.notna(row.get('HS')) else None,
                                 shots_away=int(row['AS']) if pd.notna(row.get('AS')) else None,
                                 shots_on_target_home=int(row['HST']) if pd.notna(row.get('HST')) else None,
                                 shots_on_target_away=int(row['AST']) if pd.notna(row.get('AST')) else None,
                                 corners_home=int(row['HC']) if pd.notna(row.get('HC')) else None,
                                 corners_away=int(row['AC']) if pd.notna(row.get('AC')) else None,
                                 fouls_home=int(row['HF']) if pd.notna(row.get('HF')) else None,
                                 fouls_away=int(row['AF']) if pd.notna(row.get('AF')) else None,
                                 yellow_cards_home=int(row['HY']) if pd.notna(row.get('HY')) else None,
                                 yellow_cards_away=int(row['AY']) if pd.notna(row.get('AY')) else None,
                                 red_cards_home=int(row['HR']) if pd.notna(row.get('HR')) else None,
                                 red_cards_away=int(row['AR']) if pd.notna(row.get('AR')) else None,
                             )
                         except Exception:
                             pass # Robustness

                m = Match(
                    id=mid,
                    home_team=home,
                    away_team=away,
                    league=league_enum,
                    commence_time=commence,
                    home_score=home_score,
                    away_score=away_score,
                    enrichment=row.get("_enrichment") if pd.notna(row.get("_enrichment")) else None,
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
                if "HomeTeam" not in features_df.columns and "home_team" in features_df.columns:
                    features_df["HomeTeam"] = features_df["home_team"]
                if "AwayTeam" not in features_df.columns and "away_team" in features_df.columns:
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
                    for _, row in features_df.iterrows():
                        mid = generate_match_id(
                            str(row.get(ht_col, "")),
                            str(row.get(at_col, "")),
                            row.get(dt_col),
                        )
                        mid_to_eid[mid] = str(row[eid_col])
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
            # Look for v3 weights
            v3_path = models_dir / "v3_transformer.pth"
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
        """Compute no-vig probabilities from best prices."""
        market_probs = {}
        
        # Group by event
        if "event_id" not in best_prices.columns:
            return market_probs
        
        for event_id, group in best_prices.groupby("event_id"):
            # Get implied probs
            implied = {}
            for _, row in group.iterrows():
                outcome = row.get("outcome_name", "unknown")
                price = row.get("outcome_price", 2.0)
                implied[outcome] = 1.0 / price if price > 0 else 0
            
            # Normalize (remove vig)
            total = sum(implied.values())
            if total > 0:
                market_probs[str(event_id)] = {k: v / total for k, v in implied.items()}
        
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
        
        for _, match in matches_df.iterrows():
            event_id = str(match["event_id"])
            home = match.get("home_team", "Home")
            away = match.get("away_team", "Away")
            league = match.get("league", match.get("sport_key", "unknown"))
            
            if event_id not in model_probs:
                continue
            
            event_markets = model_probs[event_id]  # {market: {outcome: prob}}
            event_market_probs = market_probs.get(event_id, {})
            
            # Get best prices for this event (handle str/int mismatch)
            event_prices = best_prices[best_prices["event_id"].astype(str) == event_id]
            
            for _, price_row in event_prices.iterrows():
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
                
                if p_model is None:
                    continue
                
                # Double Chance derivation (if missing)
                if p_model is None and market_key == "double_chance":
                     probs_1x2 = event_markets.get("1x2", {})
                     normalized_outcome = outcome.lower()
                     if "1x" in normalized_outcome or "home" in normalized_outcome:
                         p_model = (probs_1x2.get("home", 0) + probs_1x2.get("draw", 0))
                     elif "x2" in normalized_outcome or "away" in normalized_outcome:
                         p_model = (probs_1x2.get("away", 0) + probs_1x2.get("draw", 0))
                     elif "12" in normalized_outcome:
                         p_model = (probs_1x2.get("home", 0) + probs_1x2.get("away", 0))
                
                p_market = event_market_probs.get(outcome, 1.0 / odds)
                
                # Blend model and market probabilities
                p_blended = self._blender.blend(p_model, p_market, league)
                
                # EV = p * odds - 1
                ev = p_blended * odds - 1.0
                edge = p_blended - (1.0 / odds)
                
                if ev < self.config.min_ev:
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
            # Divergence filter
            if bet.divergence_level == "extreme":
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
