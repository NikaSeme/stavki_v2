"""
Deep Interaction Network - Live Feature Builder
=================================================
Transforms raw match data (API or DataFrame) into the tensor format
expected by DeepInteractionNetwork.

This is the bridge between:
  - SportMonks API / pd.DataFrame (live data)
  - StavkiDataset tensor format (model input)
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch

from stavki.config import PROJECT_ROOT

logger = logging.getLogger(__name__)


class DeepFeatureBuilder:
    """
    Builds input tensors for DeepInteractionNetwork from live match data.

    Loads Gold Layer parquets once at init, then serves fast lookups
    for real-time inference. All features are pre-match safe (no leakage).
    """

    def __init__(self, features_dir: Path = None):
        if features_dir is None:
            features_dir = PROJECT_ROOT / "data" / "processed"

        self.features_dir = features_dir

        # Player ID → Embedding Index
        self.player_map: Dict[int, int] = {}
        self.num_players: int = 0

        # Pre-loaded Gold DataFrames (indexed by team_id)
        self._team_vectors: Optional[pd.DataFrame] = None
        self._context_history: Optional[pd.DataFrame] = None
        self._momentum: Optional[pd.DataFrame] = None

        # Team ID → Latest rolling features (for live inference without match_id)
        self._team_latest_vectors: Dict[int, np.ndarray] = {}
        self._team_latest_context: Dict[int, np.ndarray] = {}
        self._team_latest_momentum: Dict[int, np.ndarray] = {}

        self._load_gold_data()

    def _load_gold_data(self):
        """Load all Gold Layer parquets and build lookup tables."""
        root = self.features_dir

        # 1. Player Map
        player_path = root / "players" / "player_features_gold.parquet"
        if player_path.exists():
            players_df = pd.read_parquet(player_path)
            unique_players = players_df['player_id'].unique()
            self.player_map = {int(pid): i + 1 for i, pid in enumerate(unique_players)}
            self.num_players = len(unique_players) + 1  # +1 for padding idx 0
            logger.info(f"DeepFeatureBuilder: Loaded {self.num_players} player embeddings")
        else:
            logger.warning(f"Player features not found at {player_path}")

        # 2. Team Vectors (latest per team)
        tv_path = root / "teams" / "team_vectors_gold.parquet"
        if tv_path.exists():
            tv = pd.read_parquet(tv_path)
            # Keep the LATEST entry per team (sorted by match_id as proxy for time)
            tv = tv.sort_values('match_id')
            for tid, group in tv.groupby('team_id'):
                row = group.iloc[-1]
                vec_cols = [c for c in row.index if c not in ('match_id', 'team_id')]
                self._team_latest_vectors[int(tid)] = row[vec_cols].values.astype(np.float32)
            logger.info(f"DeepFeatureBuilder: Loaded vectors for {len(self._team_latest_vectors)} teams")

        # 3. Context History (latest per team)
        ctx_path = root / "teams" / "context_features_gold.parquet"
        if ctx_path.exists():
            ctx = pd.read_parquet(ctx_path)
            ctx = ctx.sort_values('match_id') if 'match_id' in ctx.columns else ctx
            for tid, group in ctx.groupby('team_id'):
                row = group.iloc[-1]
                vec_cols = [c for c in row.index if c not in ('match_id', 'team_id')]
                self._team_latest_context[int(tid)] = row[vec_cols].values.astype(np.float32)
            logger.info(f"DeepFeatureBuilder: Loaded context for {len(self._team_latest_context)} teams")

        # 4. Momentum (latest per team)
        mom_path = root / "teams" / "momentum_features_gold.parquet"
        if mom_path.exists():
            mom = pd.read_parquet(mom_path)
            mom = mom.sort_values('match_id') if 'match_id' in mom.columns else mom
            for tid, group in mom.groupby('team_id'):
                row = group.iloc[-1]
                vec_cols = [c for c in row.index if c not in ('match_id', 'team_id')]
                self._team_latest_momentum[int(tid)] = row[vec_cols].values.astype(np.float32)
            logger.info(f"DeepFeatureBuilder: Loaded momentum for {len(self._team_latest_momentum)} teams")

    def get_player_indices(self, player_ids: list) -> torch.LongTensor:
        """
        Convert a list of SportMonks player IDs to embedding indices.
        Pads/truncates to exactly 11 players.
        """
        indices = [self.player_map.get(int(pid), 0) for pid in player_ids]
        if len(indices) > 11:
            indices = indices[:11]
        elif len(indices) < 11:
            indices = indices + [0] * (11 - len(indices))
        return torch.LongTensor(indices)

    def build_context_vector(self, team_id: int, is_home: bool) -> torch.FloatTensor:
        """
        Build the 13-dim context vector for a team:
        [7 XI stats] + [5 history stats] + [1 home flag]
        """
        # XI Stats (7 features)
        xi = self._team_latest_vectors.get(team_id, np.zeros(7, dtype=np.float32))

        # History Stats (5 features: red_cards, injuries, penalties, own_goals, var)
        hist = self._team_latest_context.get(team_id, np.zeros(5, dtype=np.float32))

        # Home Flag
        home_flag = np.array([1.0 if is_home else 0.0], dtype=np.float32)

        return torch.FloatTensor(np.concatenate([xi, hist, home_flag]))

    def build_momentum_vector(self, team_id: int) -> torch.FloatTensor:
        """Build the 13-dim momentum vector for a team."""
        mom = self._team_latest_momentum.get(team_id, np.zeros(13, dtype=np.float32))
        return torch.FloatTensor(mom)

    def build_match_tensors(
        self,
        home_team_id: int,
        away_team_id: int,
        home_player_ids: list = None,
        away_player_ids: list = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build the complete tensor dict for a single match.

        Args:
            home_team_id: SportMonks team ID for home team
            away_team_id: SportMonks team ID for away team
            home_player_ids: List of player IDs (starting XI). Falls back to zeros.
            away_player_ids: List of player IDs (starting XI). Falls back to zeros.

        Returns:
            Dict matching StavkiDataset.__getitem__ format (without targets)
        """
        h_players = self.get_player_indices(home_player_ids or [])
        a_players = self.get_player_indices(away_player_ids or [])

        h_ctx = self.build_context_vector(home_team_id, is_home=True)
        a_ctx = self.build_context_vector(away_team_id, is_home=False)

        h_mom = self.build_momentum_vector(home_team_id)
        a_mom = self.build_momentum_vector(away_team_id)

        return {
            'home_players': h_players,   # (11,)
            'away_players': a_players,   # (11,)
            'home_positions': torch.zeros(11, dtype=torch.long),
            'away_positions': torch.zeros(11, dtype=torch.long),
            'home_context': h_ctx,       # (13,)
            'away_context': a_ctx,       # (13,)
            'home_momentum': h_mom,      # (13,)
            'away_momentum': a_mom,      # (13,)
        }

    def get_team_id_by_name(self, team_name: str) -> Optional[int]:
        """
        Attempt to resolve a team name to its SportMonks ID.
        Uses canonical_to_id.json for an exact reverse lookup cache.
        """
        if not hasattr(self, '_name_to_id_cache'):
            self._name_to_id_cache = {}
            map_path = PROJECT_ROOT / "data" / "mapping" / "canonical_to_id.json"
            if map_path.exists():
                import json
                try:
                    with open(map_path, "r") as f:
                        self._name_to_id_cache = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load canonical_to_id.json: {e}")
                    
        # Map using TeamMapper first for canonicalization
        from stavki.data.processors.normalize import TeamMapper
        mapper = TeamMapper.get_instance()
        
        canon_name = mapper.map_name(team_name)
        if not canon_name:
            canon_name = team_name
            
        name_lower = canon_name.lower().strip()
        
        # 1. Check direct map
        if hasattr(self, '_name_to_id_cache') and name_lower in self._name_to_id_cache:
            return self._name_to_id_cache[name_lower]
            
        return None
