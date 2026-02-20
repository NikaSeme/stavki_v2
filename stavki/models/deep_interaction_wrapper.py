"""
Deep Interaction Network - Ensemble Wrapper
=============================================
Implements the BaseModel interface so the Deep Interaction Network
can be registered in EnsemblePredictor alongside CatBoost, LightGBM, etc.

Usage:
    from stavki.models.deep_interaction_wrapper import DeepInteractionWrapper
    
    model = DeepInteractionWrapper()
    model.load_checkpoint("models/deep_interaction_v3.pth")
    
    ensemble.add_model(model)
"""
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import ast
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from stavki.models.base import BaseModel, Market, Prediction
from stavki.config import PROJECT_ROOT
from stavki.utils import generate_match_id

logger = logging.getLogger(__name__)


class DeepInteractionWrapper(BaseModel):
    """
    BaseModel wrapper for DeepInteractionNetwork.
    
    Allows the Deep Network to participate in ensemble predictions
    alongside tabular models (CatBoost, LightGBM, etc.).
    
    The model uses deep features (player embeddings, attention, 
    cross-team interactions) that are fundamentally different from
    the tabular features used by other models - providing true
    diversity in the ensemble.
    """

    def __init__(self):
        super().__init__(
            name="DeepInteraction",
            markets=[Market.MATCH_WINNER],
        )
        
        self.network = None
        self.feature_builder = None
        self.device = None
        self._checkpoint_path: Optional[str] = None
        
        # Config (must match training)
        self._num_players = 0
        self._num_teams = 500
        self._num_leagues = 10
        self._num_referees = 500
        self._league_map: Dict[int, int] = {}  # SportMonks league_id → dense index
        self._team_map: Dict[int, int] = {}    # SportMonks team_id → dense index (venue)
        self._referee_map: Dict[int, int] = {} # SportMonks referee_id → dense index
        self._manager_map: Dict[int, int] = {} # SportMonks coach_id → dense index
        self._num_managers = 0
        
        # H2H Memory Cache for live inference
        self._h2h_cache: Dict[tuple, list] = {}
        
        self._embed_dim = 32
        self._league_dim = 8
        self._venue_dim = 8
        self._referee_dim = 8
        self._season_dim = 4
        self._h2h_dim = 8
        self._h2h_input_dim = 4
        self._context_dim = 13
        self._momentum_dim = 13
        self._manager_dim = 16
        self._hidden_dim = 128
        self._num_heads = 4

    def load_checkpoint(self, checkpoint_path: str = None):
        """
        Load a trained DeepInteractionNetwork checkpoint.
        
        Args:
            checkpoint_path: Path to the .pth file. 
                             Defaults to models/deep_interaction_v3.pth
        """
        if not HAS_TORCH:
            logger.error("PyTorch not available. Cannot load DeepInteractionNetwork.")
            return
        
        from stavki.models.deep_interaction import DeepInteractionNetwork
        from stavki.features.builders.deep_features import DeepFeatureBuilder
        
        # Initialize feature builder (loads Gold parquets)
        self.feature_builder = DeepFeatureBuilder()
        self._num_players = self.feature_builder.num_players
        
        if self._num_players < 2:
            logger.error("No players found in Gold data. Cannot initialize network.")
            return
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Initialize network architecture (will be re-initialized if checkpoint has different dims)
        self.network = DeepInteractionNetwork(
            num_players=self._num_players,
            num_teams=self._num_teams,
            num_leagues=self._num_leagues,
            num_referees=self._num_referees,
            embed_dim=self._embed_dim,
            league_dim=self._league_dim,
            venue_dim=self._venue_dim,
            referee_dim=self._referee_dim,
            season_dim=self._season_dim,
            h2h_dim=self._h2h_dim,
            h2h_input_dim=self._h2h_input_dim,
            context_dim=self._context_dim,
            momentum_dim=self._momentum_dim,
            manager_dim=self._manager_dim,
            hidden_dim=self._hidden_dim,
            num_heads=self._num_heads,
        ).to(self.device)
        
        # Load weights
        if checkpoint_path is None:
            checkpoint_path = str(PROJECT_ROOT / "models" / "deep_interaction_v3.pth")
        
        self._checkpoint_path = checkpoint_path
        path = Path(checkpoint_path)
        
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                
                # Handle both old (raw state_dict) and new (dict with metadata) formats
                if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    self._num_leagues = checkpoint.get('num_leagues', 10)
                    self._num_teams = checkpoint.get('num_teams', 500)
                    self._num_referees = checkpoint.get('num_referees', 500)
                    self._league_map = checkpoint.get('league_map', {})
                    self._team_map = checkpoint.get('team_map', {})
                    self._referee_map = checkpoint.get('referee_map', {})
                    self._manager_map = checkpoint.get('manager_map', {})
                    self._num_managers = checkpoint.get('num_managers', 0)
                    # Convert string keys back to int if needed (JSON serialization)
                    self._league_map = {int(k): v for k, v in self._league_map.items()}
                    self._team_map = {int(k): v for k, v in self._team_map.items()}
                    self._referee_map = {int(k): v for k, v in self._referee_map.items()}
                    self._manager_map = {int(k): v for k, v in self._manager_map.items()}
                    logger.info(f"Loaded league map: {self._league_map}")
                    logger.info(f"Loaded team map: {len(self._team_map)} teams")
                    logger.info(f"Loaded referee map: {len(self._referee_map)} referees")
                    logger.info(f"Loaded manager map: {len(self._manager_map)} managers")
                    
                    # Rebuild H2H Cache from disk
                    self._build_h2h_cache()
                    
                    # Reinitialize network with correct num_leagues if different
                    self.network = DeepInteractionNetwork(
                        num_players=self._num_players,
                        num_teams=self._num_teams,
                        num_leagues=self._num_leagues,
                        num_referees=self._num_referees,
                        num_managers=max(self._num_managers, 1),
                        embed_dim=self._embed_dim,
                        league_dim=self._league_dim,
                        venue_dim=self._venue_dim,
                        referee_dim=self._referee_dim,
                        season_dim=self._season_dim,
                        h2h_dim=self._h2h_dim,
                        h2h_input_dim=self._h2h_input_dim,
                        context_dim=self._context_dim,
                        momentum_dim=self._momentum_dim,
                        manager_dim=self._manager_dim,
                        hidden_dim=self._hidden_dim,
                        num_heads=self._num_heads,
                    ).to(self.device)
                    self.network.load_state_dict(state_dict, strict=False)
                else:
                    state_dict = checkpoint
                    self.network.load_state_dict(state_dict)
                
                self.network.eval()
                self.is_fitted = True
                
                total_params = sum(p.numel() for p in self.network.parameters())
                logger.info(
                    f"DeepInteraction loaded: {total_params:,} params, "
                    f"{self._num_players} players, {self._num_leagues} leagues, {self._num_referees} referees, device={self.device}"
                )
            except Exception as e:
                logger.error(f"Failed to load DeepInteraction from {checkpoint_path}: {e}")
                return False
                
            return True
        else:
            logger.warning(f"Checkpoint not found at {path}. Model not loaded.")
            return False

    def _build_h2h_cache(self):
        """Build H2H match history cache from Silver matches for live inference."""
        silver_path = PROJECT_ROOT / "data" / "processed" / "matches" / "matches_silver.parquet"
        if not silver_path.exists():
            logger.warning("No matches_silver.parquet found. H2H cache will be empty.")
            return
            
        logger.info("Building H2H cache from Silver matches...")
        try:
            df = pd.read_parquet(silver_path)
            # Ensure sorted by date to preserve temporal integrity
            df = df.sort_values('date')
            
            cache = {}
            for _, row in df.iterrows():
                h_id = int(row['home_team_id'])
                a_id = int(row['away_team_id'])
                pair = tuple(sorted([h_id, a_id]))
                is_first = (h_id == pair[0])
                
                outcome = int(row['outcome'])  # 0=home, 1=draw, 2=away
                goals = float(row['home_score']) + float(row['away_score'])
                
                if is_first:
                    stored_outcome = outcome
                else:
                    stored_outcome = 2 - outcome if outcome != 1 else 1
                    
                cache.setdefault(pair, []).append((stored_outcome, goals))
                
            self._h2h_cache = cache
            logger.info(f"Built H2H cache for {len(cache)} unique matchups.")
        except Exception as e:
            logger.error(f"Failed to build H2H cache: {e}")

    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Training is handled by scripts/train_deep_interaction.py.
        This method is a no-op for the wrapper (training uses a custom
        PyTorch loop, not the BaseModel.fit() interface).
        """
        logger.info(
            "DeepInteraction.fit() called — training is handled externally "
            "via scripts/train_deep_interaction.py. Use load_checkpoint() instead."
        )
        return {"note": "External training — use load_checkpoint()"}

    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """
        Generate 1X2 predictions for matches in the DataFrame.
        
        Expected columns: HomeTeam, AwayTeam, Date (or home_team, away_team)
        Optional: home_team_id, away_team_id (SportMonks IDs for direct lookup)
        
        Falls back to team name resolution if IDs not provided.
        """
        if not self.is_fitted or self.network is None:
            logger.warning("DeepInteraction not loaded. Returning empty predictions.")
            return []
        
        if self.feature_builder is None:
            logger.error("Feature builder not initialized.")
            return []
        
        predictions = []
        self.network.eval()
        
        with torch.no_grad():
            # Batching (could be optimized, but ok for now)
            for idx, row in data.iterrows():
                pred = self._predict_single(row, idx)
                if pred is not None:
                    predictions.append(pred)
        
        return predictions

    def _predict_single(self, row: pd.Series, idx) -> Optional[Prediction]:
        """Generate prediction for a single match."""
        # 1. Resolve team IDs
        home_id = self._resolve_team_id(row, is_home=True)
        away_id = self._resolve_team_id(row, is_home=False)
        
        if home_id is None or away_id is None:
            home_name = row.get('HomeTeam', row.get('home_team', '?'))
            away_name = row.get('AwayTeam', row.get('away_team', '?'))
            logger.debug(f"Could not resolve team IDs for {home_name} vs {away_name}")
            return None
        
        # 2. Get player IDs (if available from API data)
        home_players = row.get('home_player_ids', [])
        away_players = row.get('away_player_ids', [])
        
        # Handle string representation of lists
        if isinstance(home_players, str):
            try:
                home_players = ast.literal_eval(home_players)
            except (ValueError, SyntaxError):
                home_players = []
        if isinstance(away_players, str):
            try:
                away_players = ast.literal_eval(away_players)
            except (ValueError, SyntaxError):
                away_players = []
        
        # 3. Build tensors
        tensors = self.feature_builder.build_match_tensors(
            home_team_id=home_id,
            away_team_id=away_id,
            home_player_ids=home_players if home_players else None,
            away_player_ids=away_players if away_players else None,
        )
        
        # 4. Move to device
        h_p = tensors['home_players'].to(self.device).unsqueeze(0)
        a_p = tensors['away_players'].to(self.device).unsqueeze(0)
        h_pos = tensors['home_positions'].to(self.device).unsqueeze(0)
        a_pos = tensors['away_positions'].to(self.device).unsqueeze(0)
        h_c = tensors['home_context'].to(self.device).unsqueeze(0)
        a_c = tensors['away_context'].to(self.device).unsqueeze(0)
        h_m = tensors['home_momentum'].to(self.device).unsqueeze(0)
        a_m = tensors['away_momentum'].to(self.device).unsqueeze(0)
        
        # 5. Resolve league_id
        raw_league_id = row.get('league_id', row.get('League_id', 0))
        if isinstance(raw_league_id, (int, float)) and int(raw_league_id) in self._league_map:
            dense_league = self._league_map[int(raw_league_id)]
        else:
            dense_league = 0
        league_tensor = torch.tensor([dense_league], dtype=torch.long).to(self.device)
        
        # 5b. Resolve venue_id (home_team_id → dense index)
        if home_id in self._team_map:
            dense_venue = self._team_map[home_id]
        else:
            dense_venue = 0
        venue_tensor = torch.tensor([dense_venue], dtype=torch.long).to(self.device)
        
        # 5c. Resolve season phase from date
        date_str = row.get('Date', row.get('date', ''))
        try:
            month = pd.to_datetime(date_str).month
            # European football calendar phases
            MONTH_MAP = {8:0,9:0,10:0, 11:1,12:1,1:1,2:1, 3:2,4:2,5:2, 6:3,7:3}
            season_phase = MONTH_MAP.get(month, 0)
        except Exception:
            season_phase = 0
        season_tensor = torch.tensor([season_phase], dtype=torch.long).to(self.device)
        
        # 5d. Resolve referee_id
        raw_referee_id = row.get('referee_id', 0)
        if isinstance(raw_referee_id, (int, float)) and int(raw_referee_id) in self._referee_map:
            dense_referee = self._referee_map[int(raw_referee_id)]
        else:
            dense_referee = 0
        referee_tensor = torch.tensor([dense_referee], dtype=torch.long).to(self.device)
        
        # 5f. Resolve Managers
        raw_home_coach = row.get('home_coach_id', 0)
        if int(raw_home_coach) in self._manager_map:
            dense_h_man = self._manager_map[int(raw_home_coach)]
        else:
            dense_h_man = 0
            
        raw_away_coach = row.get('away_coach_id', 0)
        if int(raw_away_coach) in self._manager_map:
            dense_a_man = self._manager_map[int(raw_away_coach)]
        else:
            dense_a_man = 0
            
        h_man_tensor = torch.tensor([dense_h_man], dtype=torch.long).to(self.device)
        a_man_tensor = torch.tensor([dense_a_man], dtype=torch.long).to(self.device)
        
        # 5e. Compute H2H features from cache
        h2h_features = [0.0, 0.0, 0.0, 0.0]  # Default (no history)
        pair = tuple(sorted([home_id, away_id]))
        history = self._h2h_cache.get(pair, [])
        
        if len(history) >= 1:
            outcomes = [h[0] for h in history]
            goals = [h[1] for h in history]
            n = len(history)
            
            is_first = (home_id == pair[0])
            if is_first:
                wins = sum(1 for o in outcomes if o == 0)
            else:
                wins = sum(1 for o in outcomes if o == 2)
            draws = sum(1 for o in outcomes if o == 1)
            
            h2h_features[0] = wins / n                       # win_rate
            h2h_features[1] = draws / n                      # draw_rate
            h2h_features[2] = np.mean(goals) / 5.0           # avg_goals (normalized)
            h2h_features[3] = np.log1p(n) / np.log1p(10)     # familiarity
            
        h2h_tensor = torch.tensor([h2h_features], dtype=torch.float32).to(self.device)
        
        # 6. Forward pass
        logits, lam_h, lam_a = self.network(
            h_p, a_p, h_pos, a_pos, h_man_tensor, a_man_tensor,
            h_c, a_c, h_m, a_m,
            league_tensor, venue_tensor, referee_tensor, season_tensor, h2h_tensor
        )
        
        # 7. Convert logits to probabilities
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
        # Clip to prevent extreme values
        probs = np.clip(probs, 0.02, 0.96)
        # Normalize after clipping
        probs = probs / probs.sum()
        
        # 8. Build match ID
        home_name = row.get('HomeTeam', row.get('home_team', str(home_id)))
        away_name = row.get('AwayTeam', row.get('away_team', str(away_id)))
        date = row.get('Date', row.get('date', ''))
        
        try:
            match_id = generate_match_id(home_name, away_name, date)
        except Exception:
            match_id = f"{home_name}_vs_{away_name}_{idx}"
        
        # 9. Build Prediction object
        return Prediction(
            match_id=match_id,
            market=Market.MATCH_WINNER,
            probabilities={
                "home": float(probs[0]),
                "draw": float(probs[1]),
                "away": float(probs[2]),
            },
            confidence=float(probs.max()),
            model_name=self.name,
        )

    def _resolve_team_id(self, row: pd.Series, is_home: bool) -> Optional[int]:
        """Resolve a team's SportMonks ID from the row data."""
        # Try direct ID columns first
        if is_home:
            for col in ('home_team_id', 'HomeTeamId', 'home_id'):
                if col in row.index and pd.notna(row[col]):
                    return int(row[col])
        else:
            for col in ('away_team_id', 'AwayTeamId', 'away_id'):
                if col in row.index and pd.notna(row[col]):
                    return int(row[col])
        
        # Fallback: resolve by name
        name_col = 'HomeTeam' if is_home else 'AwayTeam'
        alt_col = 'home_team' if is_home else 'away_team'
        
        team_name = row.get(name_col, row.get(alt_col))
        if team_name and self.feature_builder:
            return self.feature_builder.get_team_id_by_name(str(team_name))
        
        return None

    def _get_state(self) -> Dict[str, Any]:
        """Serialize model state for BaseModel.save()."""
        return {
            "checkpoint_path": self._checkpoint_path,
            "num_players": self._num_players,
            "num_teams": self._num_teams,
            "num_leagues": self._num_leagues,
            "num_referees": self._num_referees,
            "league_map": self._league_map,
            "team_map": self._team_map,
            "referee_map": self._referee_map,
            "embed_dim": self._embed_dim,
            "league_dim": self._league_dim,
            "venue_dim": self._venue_dim,
            "referee_dim": self._referee_dim,
            "season_dim": self._season_dim,
            "h2h_dim": self._h2h_dim,
            "h2h_input_dim": self._h2h_input_dim,
            "context_dim": self._context_dim,
            "momentum_dim": self._momentum_dim,
            "hidden_dim": self._hidden_dim,
            "num_heads": self._num_heads,
        }

    def _set_state(self, state: Dict[str, Any]):
        """Restore model state from BaseModel.load()."""
        self._checkpoint_path = state.get("checkpoint_path")
        self._num_players = state.get("num_players", 0)
        self._num_teams = state.get("num_teams", 500)
        self._num_leagues = state.get("num_leagues", 10)
        self._league_map = state.get("league_map", {})
        self._team_map = state.get("team_map", {})
        self._embed_dim = state.get("embed_dim", 32)
        self._league_dim = state.get("league_dim", 8)
        self._venue_dim = state.get("venue_dim", 8)
        self._season_dim = state.get("season_dim", 4)
        self._h2h_dim = state.get("h2h_dim", 8)
        self._h2h_input_dim = state.get("h2h_input_dim", 4)
        self._context_dim = state.get("context_dim", 13)
        self._momentum_dim = state.get("momentum_dim", 13)
        self._manager_dim = state.get("manager_dim", 16)
        self._hidden_dim = state.get("hidden_dim", 128)
        self._num_heads = state.get("num_heads", 4)
        
        # Reload the checkpoint if path was saved
        if self._checkpoint_path:
            self.load_checkpoint(self._checkpoint_path)
