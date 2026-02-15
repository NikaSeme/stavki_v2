"""
Live Prediction Pipeline.

Integrates SportMonks API for real-time fixture data, odds, and predictions.
Uses unified data loader for consistent data handling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

from stavki.config import PROJECT_ROOT, DATA_DIR, MODELS_DIR
from stavki.data.loader import UnifiedDataLoader, get_loader
from stavki.data.collectors.sportmonks import SportMonksClient, MatchFixture
from stavki.features.builders.formation import FormationFeatureBuilder

logger = logging.getLogger(__name__)


@dataclass
class LivePrediction:
    """Prediction for an upcoming match."""
    fixture_id: int
    home_team: str
    away_team: str
    league: str
    kickoff: datetime
    
    # Probabilities
    prob_home: float
    prob_draw: float
    prob_away: float
    
    # Odds (from API)
    odds_home: Optional[float] = None
    odds_draw: Optional[float] = None
    odds_away: Optional[float] = None
    
    # Value analysis
    ev_home: Optional[float] = None
    ev_draw: Optional[float] = None
    ev_away: Optional[float] = None
    best_bet: Optional[str] = None
    best_ev: Optional[float] = None
    recommended: bool = False
    stake_pct: Optional[float] = None


class LivePredictor:
    """
    Real-time prediction engine using SportMonks API.
    
    Usage:
        predictor = LivePredictor(api_key="...")
        predictor.load_model('/path/to/model.pkl')
        predictions = predictor.predict_upcoming(days=3)
    """
    
    # League ID mapping
    LEAGUE_MAP = {
        8: 'epl',
        82: 'bundesliga', 
        564: 'laliga',
        384: 'seriea',
        301: 'ligue1',
        9: 'championship',
    }
    
    def __init__(
        self,
        api_key: str,
        min_ev: float = 0.03,
        min_edge: float = 0.02,
        model_alpha: float = 0.55,
        kelly_fraction: float = 0.25
    ):
        self.client = SportMonksClient(api_key)
        self.min_ev = min_ev
        self.min_edge = min_edge
        self.model_alpha = model_alpha
        self.kelly_fraction = kelly_fraction
        
        self.model = None
        self.feature_cols: List[str] = []
        self.elo_ratings: Dict[str, float] = {}
        self.team_form: Dict[str, List[float]] = {}
        self.team_synth_xg: Dict[str, float] = {}  # team -> rolling synth_xg
        self.team_avg_rating: Dict[str, float] = {}  # team -> rolling avg XI rating
        self.referee_profiles: Dict[str, Dict] = {}  # referee -> profile
        
        # Phase 3: Rolling match stats per team
        self.team_fouls: Dict[str, float] = {}
        self.team_yellows: Dict[str, float] = {}
        self.team_corners: Dict[str, float] = {}
        self.team_possession: Dict[str, float] = {}
        
        # Phase 3: Referee target encoding
        self.ref_encoded_goals: Dict[str, float] = {}
        self.ref_encoded_cards: Dict[str, float] = {}
        
        # Phase 3: Formation Builder
        self.formation_builder = FormationFeatureBuilder()
        
        # Load ELO and form from historical data
        self._load_team_stats()
    
    def _load_team_stats(self):
        """Load current ELO and form from historical data."""
        try:
            features_path = DATA_DIR / 'features_full.parquet'
            if features_path.exists():
                df = pd.read_parquet(features_path)
            else:
                df = pd.read_csv(DATA_DIR / 'features_full.csv', low_memory=False)
            
            # Get latest ELO for each team
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.sort_values('Date')
            
            # Extract ELO from latest matches
            for _, row in df.tail(1000).iterrows():
                home = row.get('HomeTeam', '')
                away = row.get('AwayTeam', '')
                
                if home and 'elo_home' in row:
                    self.elo_ratings[home] = row['elo_home']
                if away and 'elo_away' in row:
                    self.elo_ratings[away] = row['elo_away']
                    
                # Form (last points)
                if home and 'form_home_pts' in row:
                    if home not in self.team_form:
                        self.team_form[home] = []
                    self.team_form[home].append(row['form_home_pts'])
                    self.team_form[home] = self.team_form[home][-5:]
                    
            logger.info(f"Loaded ELO for {len(self.elo_ratings)} teams")
            
            # Load Tier 1 features from enriched CSV if available
            enriched_path = DATA_DIR / 'features_full_enriched.csv'
            src = enriched_path if enriched_path.exists() else features_path if features_path.exists() else DATA_DIR / 'features_full.csv'
            if src.suffix == '.csv':
                edf = pd.read_csv(src, low_memory=False)
            else:
                edf = pd.read_parquet(src)
            edf = edf.sort_values('Date') if 'Date' in edf.columns else edf
            
            # Rolling synth_xg and ratings from last 1000 rows
            for _, row in edf.tail(1000).iterrows():
                home = row.get('HomeTeam', '')
                away = row.get('AwayTeam', '')
                if home and 'synth_xg_home' in row and pd.notna(row['synth_xg_home']):
                    self.team_synth_xg[home] = float(row['synth_xg_home'])
                if away and 'synth_xg_away' in row and pd.notna(row['synth_xg_away']):
                    self.team_synth_xg[away] = float(row['synth_xg_away'])
                if home and 'avg_rating_home' in row and pd.notna(row['avg_rating_home']):
                    self.team_avg_rating[home] = float(row['avg_rating_home'])
                if away and 'avg_rating_away' in row and pd.notna(row['avg_rating_away']):
                    self.team_avg_rating[away] = float(row['avg_rating_away'])
            
            # Build referee profiles from Referee column
            if 'Referee' in edf.columns:
                from collections import defaultdict
                ref_data = defaultdict(lambda: {'matches': 0, 'goals': 0, 'cards': 0, 'over25': 0})
                for _, row in edf.iterrows():
                    ref = row.get('Referee')
                    if pd.isna(ref) or not ref:
                        continue
                    ref = str(ref).strip().lower()
                    rd = ref_data[ref]
                    rd['matches'] += 1
                    goals = (row.get('FTHG', 0) or 0) + (row.get('FTAG', 0) or 0)
                    rd['goals'] += goals
                    if goals > 2:
                        rd['over25'] += 1
                    rd['cards'] += (row.get('HY', 0) or 0) + (row.get('AY', 0) or 0) + \
                                   (row.get('HR', 0) or 0) + (row.get('AR', 0) or 0)
                for ref, rd in ref_data.items():
                    n = rd['matches']
                    if n >= 5:
                        self.referee_profiles[ref] = {
                            'goals_pg': round(rd['goals'] / n, 2),
                            'cards_pg': round(rd['cards'] / n, 2),
                            'over25_rate': round(rd['over25'] / n, 3),
                            'strictness': round((rd['cards']/n - 3.5) / 1.5, 3),
                        }
            
            logger.info(f"Loaded synth_xg for {len(self.team_synth_xg)} teams, "
                        f"ratings for {len(self.team_avg_rating)} teams, "
                        f"{len(self.referee_profiles)} referee profiles")
            
            # Phase 3: Load rolling match stats from enriched data
            for _, row in edf.tail(1000).iterrows():
                home = row.get('HomeTeam', '')
                away = row.get('AwayTeam', '')
                # Rolling fouls
                if home and 'rolling_fouls_home' in row and pd.notna(row['rolling_fouls_home']):
                    self.team_fouls[home] = float(row['rolling_fouls_home'])
                if away and 'rolling_fouls_away' in row and pd.notna(row['rolling_fouls_away']):
                    self.team_fouls[away] = float(row['rolling_fouls_away'])
                # Rolling yellows
                if home and 'rolling_yellows_home' in row and pd.notna(row['rolling_yellows_home']):
                    self.team_yellows[home] = float(row['rolling_yellows_home'])
                if away and 'rolling_yellows_away' in row and pd.notna(row['rolling_yellows_away']):
                    self.team_yellows[away] = float(row['rolling_yellows_away'])
                # Rolling corners
                if home and 'rolling_corners_home' in row and pd.notna(row['rolling_corners_home']):
                    self.team_corners[home] = float(row['rolling_corners_home'])
                if away and 'rolling_corners_away' in row and pd.notna(row['rolling_corners_away']):
                    self.team_corners[away] = float(row['rolling_corners_away'])
                # Rolling possession
                if home and 'rolling_possession_home' in row and pd.notna(row['rolling_possession_home']):
                    self.team_possession[home] = float(row['rolling_possession_home'])
                if away and 'rolling_possession_away' in row and pd.notna(row['rolling_possession_away']):
                    self.team_possession[away] = float(row['rolling_possession_away'])
            
            # Phase 3: Referee target encoding from enriched data 
            if 'ref_encoded_goals' in edf.columns and 'Referee' in edf.columns:
                for _, row in edf.tail(5000).iterrows():
                    ref = row.get('Referee')
                    if pd.isna(ref) or not ref:
                        continue
                    ref = str(ref).strip().lower()
                    if 'ref_encoded_goals' in row and pd.notna(row['ref_encoded_goals']):
                        self.ref_encoded_goals[ref] = float(row['ref_encoded_goals'])
                    if 'ref_encoded_cards' in row and pd.notna(row['ref_encoded_cards']):
                        self.ref_encoded_cards[ref] = float(row['ref_encoded_cards'])
            
            # Phase 3: Load historical formations for builder
            # Use columns formation_home / formation_away if available
            if 'formation_home' in edf.columns and 'formation_away' in edf.columns:
                 # Manually populate builder's internal state
                 team_fmts = self.formation_builder._team_formations
                 for _, row in edf.tail(2000).iterrows():
                     h_name = str(row.get("HomeTeam")).strip().lower()
                     a_name = str(row.get("AwayTeam")).strip().lower()
                     h_fmt = row.get("formation_home")
                     a_fmt = row.get("formation_away")
                     
                     if h_name and pd.notna(h_fmt):
                         if h_name not in team_fmts: team_fmts[h_name] = []
                         team_fmts[h_name].append(str(h_fmt))
                         team_fmts[h_name] = team_fmts[h_name][-10:]
                         
                     if a_name and pd.notna(a_fmt):
                         if a_name not in team_fmts: team_fmts[a_name] = []
                         team_fmts[a_name].append(str(a_fmt))
                         team_fmts[a_name] = team_fmts[a_name][-10:]
                 self.formation_builder._is_fitted = True

            logger.info(f"Phase 3: {len(self.team_fouls)} teams with fouls, "
                        f"{len(self.team_corners)} with corners, "
                        f"{len(self.ref_encoded_goals)} refs encoded, "
                        f"{len(self.formation_builder._team_formations)} teams with formations")
            
        except Exception as e:
            logger.warning(f"Could not load team stats: {e}")
    
    def load_model(self, model_path: str):
        """Load trained CatBoost model."""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data.get('model')
        self.feature_cols = data.get('features', [])
        logger.info(f"Loaded model with {len(self.feature_cols)} features")
    
    def _build_features(self, fixture: MatchFixture, odds: Dict) -> pd.DataFrame:
        """Build feature vector for a fixture."""
        home = fixture.home_team
        away = fixture.away_team
        
        # ELO features
        elo_home = self.elo_ratings.get(home, 1500)
        elo_away = self.elo_ratings.get(away, 1500)
        elo_diff = elo_home - elo_away
        
        # Form features
        form_home = self.team_form.get(home, [7.5])
        form_away = self.team_form.get(away, [7.5])
        form_home_pts = np.mean(form_home) if form_home else 7.5
        form_away_pts = np.mean(form_away) if form_away else 7.5
        form_diff = form_home_pts - form_away_pts
        
        # Odds features
        odds_h = odds.get('home', 2.5)
        odds_d = odds.get('draw', 3.3)
        odds_a = odds.get('away', 2.8)
        
        margin = 1/odds_h + 1/odds_d + 1/odds_a
        imp_home = (1/odds_h) / margin
        imp_draw = (1/odds_d) / margin
        imp_away = (1/odds_a) / margin
        
        # Tier 1: Synthetic xG
        synth_xg_home = self.team_synth_xg.get(home, 1.3)
        synth_xg_away = self.team_synth_xg.get(away, 1.1)
        
        # Tier 1: Player Ratings
        avg_rating_home = self.team_avg_rating.get(home, 6.5)
        avg_rating_away = self.team_avg_rating.get(away, 6.5)
        
        # Tier 1: Referee features
        # Try to get referee from fixture data (if available from API)
        ref_name = getattr(fixture, 'referee', None)
        if ref_name:
            ref_name = str(ref_name).strip().lower()
        ref = self.referee_profiles.get(ref_name, {}) if ref_name else {}
        
        features = {
            'elo_diff': elo_diff,
            'elo_home': elo_home,
            'elo_away': elo_away,
            'form_diff': form_diff,
            'form_home_pts': form_home_pts,
            'form_away_pts': form_away_pts,
            'gf_diff': 0,  # Not available live
            'ga_diff': 0,
            'B365H': odds_h,
            'B365D': odds_d,
            'B365A': odds_a,
            'imp_home_norm': imp_home,
            'imp_draw_norm': imp_draw,
            'imp_away_norm': imp_away,
            # Tier 1: Synthetic xG
            'synth_xg_home': synth_xg_home,
            'synth_xg_away': synth_xg_away,
            'synth_xg_diff': round(synth_xg_home - synth_xg_away, 3),
            # Tier 1: Player Ratings
            'avg_rating_home': avg_rating_home,
            'avg_rating_away': avg_rating_away,
            'rating_delta': round(avg_rating_home - avg_rating_away, 2),
            # Tier 1: Referee
            'ref_goals_per_game': ref.get('goals_pg', 2.7),
            'ref_cards_per_game_t1': ref.get('cards_pg', 3.5),
            'ref_over25_rate': ref.get('over25_rate', 0.55),
            'ref_strictness_t1': ref.get('strictness', 0.0),
        }
        
        # Phase 3: Formation features
        from stavki.data.schemas import Match, Team, League
        
        # Construct minimal Match object for the builder
        match_obj = Match(
            id=str(fixture.fixture_id),
            home_team=Team(name=home),
            away_team=Team(name=away),
            league=League.EPL, # Dummy valid league, builder doesn't use it
            commence_time=fixture.kickoff,
            source="live_predictor"
        )
        
        fmt_features = self.formation_builder.get_features(match_obj)
        features.update(fmt_features)
        
        # Phase 3: Rolling match stats & Ref encodings
        features.update({
            'rolling_fouls_home': self.team_fouls.get(home, 12.0),
            'rolling_fouls_away': self.team_fouls.get(away, 12.0),
            'rolling_yellows_home': self.team_yellows.get(home, 2.0),
            'rolling_yellows_away': self.team_yellows.get(away, 2.0),
            'rolling_corners_home': self.team_corners.get(home, 5.0),
            'rolling_corners_away': self.team_corners.get(away, 5.0),
            'rolling_possession_home': self.team_possession.get(home, 50.0),
            'rolling_possession_away': self.team_possession.get(away, 50.0),
            'ref_encoded_goals': self.ref_encoded_goals.get(ref_name, 2.7) if ref_name else 2.7,
            'ref_encoded_cards': self.ref_encoded_cards.get(ref_name, 3.9) if ref_name else 3.9,
        })
        
        return pd.DataFrame([features])
    
    def predict_fixture(
        self,
        fixture: MatchFixture,
        odds: Optional[Dict] = None
    ) -> LivePrediction:
        """
        Generate prediction for a single fixture.
        
        Args:
            fixture: MatchFixture from SportMonks
            odds: Optional odds dict {home, draw, away}
        """
        # Get odds from API if not provided
        if odds is None:
            api_odds = self.client.get_fixture_odds(fixture.fixture_id)
            if api_odds:
                odds = api_odds[0].get('odds', {})
            else:
                odds = {}  # No odds available
        
        # Build features
        X = self._build_features(fixture, odds)
        
        # Predict
        if self.model is not None:
            available_cols = [c for c in self.feature_cols if c in X.columns]
            X_pred = X[available_cols].fillna(0)
            
            probs = self.model.predict_proba(X_pred)[0]
            # CatBoost order: A=0, D=1, H=2
            model_probs = np.array([probs[2], probs[1], probs[0]])
        else:
            # Use ELO-based probs if no model
            elo_diff = X['elo_diff'].iloc[0]
            exp_home = 1 / (1 + 10**(-elo_diff/400))
            model_probs = np.array([exp_home, 0.27, 1 - exp_home - 0.27])
        
        # Blend with market
        imp_probs = np.array([
            X['imp_home_norm'].iloc[0],
            X['imp_draw_norm'].iloc[0], 
            X['imp_away_norm'].iloc[0]
        ])
        
        # If any market prob is NaN (missing odds), fallback to pure model
        if np.isnan(imp_probs).any():
            blended = model_probs
        else:
            blended = self.model_alpha * model_probs + (1 - self.model_alpha) * imp_probs
            
        blended = blended / blended.sum()
        
        # Expected values
        # Careful: odds might be missing
        odds_h = odds.get('home')
        odds_d = odds.get('draw')
        odds_a = odds.get('away')
        
        if odds_h and odds_d and odds_a:
            odds_arr = np.array([odds_h, odds_d, odds_a])
            evs = blended * odds_arr - 1
            
            # Find best bet
            best_idx = np.argmax(evs)
            best_ev = evs[best_idx]
            edge = blended[best_idx] - imp_probs[best_idx]
            
            # Recommended?
            recommended = best_ev >= self.min_ev and edge >= self.min_edge
            best_bet_label = ['Home', 'Draw', 'Away'][best_idx]
        else:
            evs = np.array([None, None, None])
            best_ev = None
            edge = None
            recommended = False
            best_bet_label = None
            
        # Kelly stake
        stake_pct = None
        if recommended and best_ev is not None:
            # Kelly: f* = (p*b - 1) / (b - 1), where p=prob, b=odds
            p = blended[best_idx]
            b = odds_arr[best_idx]
            if b > 1:
                kelly = (p * b - 1) / (b - 1)
                stake_pct = max(0, min(kelly * self.kelly_fraction, 0.05))
        
        return LivePrediction(
            fixture_id=fixture.fixture_id,
            home_team=fixture.home_team,
            away_team=fixture.away_team,
            league=self.LEAGUE_MAP.get(fixture.league_id, 'unknown'),
            kickoff=fixture.kickoff,
            prob_home=blended[0],
            prob_draw=blended[1],
            prob_away=blended[2],
            odds_home=odds.get('home'),
            odds_draw=odds.get('draw'),
            odds_away=odds.get('away'),
            ev_home=evs[0],
            ev_draw=evs[1],
            ev_away=evs[2],
            best_bet=best_bet_label,
            best_ev=best_ev,
            recommended=recommended,
            stake_pct=stake_pct
        )
    
    def predict_upcoming(
        self,
        days: int = 7,
        leagues: Optional[List[int]] = None
    ) -> List[LivePrediction]:
        """
        Get predictions for upcoming fixtures.
        
        Args:
            days: Number of days ahead to predict
            leagues: League IDs to include (default: all supported)
            
        Returns:
            List of LivePrediction objects
        """
        if leagues is None:
            leagues = list(self.LEAGUE_MAP.keys())
        
        # Fetch fixtures
        fixtures = self.client.get_upcoming_fixtures(days=days, league_ids=leagues)
        logger.info(f"Found {len(fixtures)} upcoming fixtures")
        
        predictions = []
        for fixture in fixtures:
            try:
                pred = self.predict_fixture(fixture)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict {fixture.home_team} vs {fixture.away_team}: {e}")
        
        # Sort by EV
        predictions.sort(key=lambda p: p.best_ev or 0, reverse=True)
        
        return predictions
    
    def get_recommendations(
        self,
        days: int = 7,
        max_bets: int = 10
    ) -> List[LivePrediction]:
        """Get top betting recommendations."""
        all_preds = self.predict_upcoming(days=days)
        recommended = [p for p in all_preds if p.recommended]
        return recommended[:max_bets]
    
    def format_predictions(self, predictions: List[LivePrediction]) -> str:
        """Format predictions as readable text."""
        if not predictions:
            return "No predictions available."
        
        lines = ["📊 LIVE PREDICTIONS", "=" * 50]
        
        for pred in predictions:
            rec = "⭐" if pred.recommended else "  "
            lines.append(f"\n{rec} {pred.home_team} vs {pred.away_team}")
            lines.append(f"   {pred.league.upper()} | {pred.kickoff.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"   Probs: {pred.prob_home:.1%}/{pred.prob_draw:.1%}/{pred.prob_away:.1%}")
            
            if pred.odds_home:
                lines.append(f"   Odds: {pred.odds_home:.2f}/{pred.odds_draw:.2f}/{pred.odds_away:.2f}")
            
            if pred.best_ev:
                lines.append(f"   Best: {pred.best_bet} @ EV={pred.best_ev:+.1%}")
            
            if pred.stake_pct:
                lines.append(f"   Stake: {pred.stake_pct:.1%} of bankroll")
        
        return "\n".join(lines)


if __name__ == "__main__":
    import os
    
    # Get API key
    env_path = PROJECT_ROOT / '.env'
    api_key = None
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith('SPORTMONKS_API_KEY='):
                    api_key = line.strip().split('=', 1)[1].strip('"').strip("'")
    
    if not api_key:
        print("❌ SPORTMONKS_API_KEY not found")
        exit(1)
    
    print("🚀 Testing Live Prediction Pipeline")
    print()
    
    predictor = LivePredictor(api_key=api_key)
    print(f"✅ Loaded ELO for {len(predictor.elo_ratings)} teams")
    
    # Get upcoming fixtures
    print()
    print("Fetching upcoming fixtures...")
    preds = predictor.predict_upcoming(days=3, leagues=[8, 82])  # EPL + Bundesliga
    
    print(predictor.format_predictions(preds[:5]))
    
    # Show recommendations
    recs = [p for p in preds if p.recommended]
    print()
    print(f"⭐ {len(recs)} recommended bets")
