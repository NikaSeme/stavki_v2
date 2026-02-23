
import logging
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from stavki.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Season phase from month (European football calendar)
MONTH_TO_SEASON_PHASE = {
    8: 0, 9: 0, 10: 0,          # Early season (Aug-Oct)
    11: 1, 12: 1, 1: 1, 2: 1,   # Mid season (Nov-Feb)
    3: 2, 4: 2, 5: 2,            # Late season (Mar-May) — relegation battles, title races
    6: 3, 7: 3,                   # Summer (Jun-Jul) — cups, international)
}
NUM_SEASON_PHASES = 4


class StavkiDataset(Dataset):
    def __init__(self, features_dir: Path = None, inference_mode=False):
        """
        PyTorch Dataset for Deep Interaction Network.
        
        Args:
            features_dir (Path): Directory containing processed parquets.
            inference_mode (bool): If True, loads only feature data (no targets).
        """
        if features_dir is None:
            features_dir = PROJECT_ROOT / "data" / "processed"
            
        self.inference_mode = inference_mode
        self.player_map = {} # ID -> Index
        self.num_players = 0
        
        # Load Data
        self._load_data(features_dir)
        
    def _load_data(self, root: Path):
        # 1. Matches (Targets)
        match_path = root / "matches" / "matches_silver.parquet"
        if not match_path.exists():
            raise FileNotFoundError(f"Matches not found at {match_path}")
        self.matches = pd.read_parquet(match_path)
        
        # 2. Player Features (Embeddings Input)
        player_path = root / "players" / "player_features_gold.parquet"
        self.players = pd.read_parquet(player_path)
        
        # Build Player Map
        unique_players = self.players['player_id'].unique()
        self.player_map = {pid: i+1 for i, pid in enumerate(unique_players)} # 0 = Padding
        self.num_players = len(unique_players) + 1
        logger.info(f"Loaded {self.num_players} unique players.")
        
        # 3. Team Features (Context)
        team_path = root / "teams" / "team_vectors_gold.parquet"
        self.team_vectors = pd.read_parquet(team_path)
        
        # 3b. Referees (Silver)
        ref_path = root / "matches" / "referees_silver.parquet"
        if ref_path.exists():
            self.referees = pd.read_parquet(ref_path)
            unique_refs = self.referees['referee_id'].dropna().unique()
            self.ref_map = {rid: i+1 for i, rid in enumerate(unique_refs)} # 0 = Unknown/Padding
            self.num_referees = len(unique_refs) + 1
            logger.info(f"Loaded {self.num_referees} unique referees.")
            
            # Map match_id -> dense referee_id
            self.match_ref_lookup = {}
            for row in self.referees.itertuples():
                if pd.notna(row.referee_id):
                    self.match_ref_lookup[row.match_id] = self.ref_map.get(int(row.referee_id), 0)
        else:
            self.referees = pd.DataFrame()
            self.ref_map = {}
            self.num_referees = 1
            self.match_ref_lookup = {}
            logger.warning("Referees data not found, defaulting to 0 padding.")
            
        # 3c. Managers (Silver Tactical Data)
        man_path = root / "matches" / "managers_silver.parquet"
        if man_path.exists():
            self.managers = pd.read_parquet(man_path)
            unique_mans = pd.concat([self.managers['home_coach_id'], self.managers['away_coach_id']]).dropna().unique()
            self.man_map = {mid: i+1 for i, mid in enumerate(unique_mans)} # 0 = Unknown/Padding
            self.num_managers = len(unique_mans) + 1
            logger.info(f"Loaded {self.num_managers} unique managers.")
            
            # Map match_id -> dense home_manager, away_manager
            self.match_home_man_lookup = {}
            self.match_away_man_lookup = {}
            for row in self.managers.itertuples():
                if pd.notna(row.home_coach_id):
                    self.match_home_man_lookup[row.match_id] = self.man_map.get(int(row.home_coach_id), 0)
                if pd.notna(row.away_coach_id):
                    self.match_away_man_lookup[row.match_id] = self.man_map.get(int(row.away_coach_id), 0)
        else:
            self.managers = pd.DataFrame()
            self.man_map = {}
            self.num_managers = 1
            self.match_home_man_lookup = {}
            self.match_away_man_lookup = {}
            logger.warning("Managers data not found, defaulting to 0 padding.")
        
        # 4. Momentum (Trends)
        mom_path = root / "teams" / "momentum_features_gold.parquet"
        self.momentum = pd.read_parquet(mom_path) if mom_path.exists() else pd.DataFrame()
        
        # 5. Context History (NLP Events)
        ctx_path = root / "teams" / "context_features_gold.parquet"
        self.context_history = pd.read_parquet(ctx_path) if ctx_path.exists() else pd.DataFrame()
        
        # Prepare Match-Level Data
        self.data_rows = []
        
        # Filter matches where we have data
        valid_matches = self.matches[self.matches['match_id'].isin(self.players['match_id'].unique())]
        
        self.match_ids = valid_matches['match_id'].values
        self.home_ids = valid_matches['home_team_id'].values
        self.away_ids = valid_matches['away_team_id'].values
        self.dates = valid_matches['date'].values
        
        # Build League Map (sparse SportMonks IDs → dense 0..N indices)
        self.league_ids_raw = valid_matches['league_id'].values
        unique_leagues = sorted(set(self.league_ids_raw))
        self.league_map = {lid: i for i, lid in enumerate(unique_leagues)}
        self.num_leagues = len(unique_leagues)
        self.league_ids = np.array([self.league_map[lid] for lid in self.league_ids_raw], dtype=np.int64)
        logger.info(f"Loaded {self.num_leagues} unique leagues: {unique_leagues}")
        
        # Build Team Map (for venue embedding: home_team_id → venue)
        all_team_ids = sorted(set(self.home_ids.tolist() + self.away_ids.tolist()))
        self.team_map = {int(tid): i + 1 for i, tid in enumerate(all_team_ids)}  # 0 = padding
        self.num_teams = len(all_team_ids) + 1
        self.venue_ids = np.array([self.team_map.get(int(tid), 0) for tid in self.home_ids], dtype=np.int64)
        logger.info(f"Loaded {self.num_teams} unique teams (venue proxy)")
        
        # Build Season Phase from date month
        date_months = pd.to_datetime(valid_matches['date']).dt.month.values
        self.season_phases = np.array(
            [MONTH_TO_SEASON_PHASE.get(m, 0) for m in date_months], dtype=np.int64
        )
        logger.info(f"Season phases built: {dict(zip(*np.unique(self.season_phases, return_counts=True)))}")
        
        # Build Head-to-Head features (temporal: only past meetings count)
        self._build_h2h_features(valid_matches)
        
        if not self.inference_mode:
            self.outcomes = valid_matches['outcome'].values.astype(np.int64)
            self.home_goals = valid_matches['home_score'].values.astype(np.float32)
            self.away_goals = valid_matches['away_score'].values.astype(np.float32)
            
        # Optimization: Pre-group players & positions
        self.match_player_lookup = {}
        self.match_position_lookup = {}
        starters = self.players[self.players['lineup_type_id'] == 11]
        
        for mid, group in starters.groupby('match_id'):
            self.match_player_lookup[mid] = {}
            self.match_position_lookup[mid] = {}
            for tid, tgroup in group.groupby('team_id'):
                pids = tgroup['player_id'].map(self.player_map).fillna(0).astype(int).tolist()
                
                # Positions (raw SportMonks position IDs, usually small ints < 50)
                pos_col = 'position_id' if 'position_id' in tgroup.columns else 'formation_position'
                pos_ids = tgroup[pos_col].fillna(0).astype(int).tolist()
                
                if len(pids) > 11: 
                    pids = pids[:11]
                    pos_ids = pos_ids[:11]
                elif len(pids) < 11: 
                    pids = pids + [0]*(11-len(pids))
                    pos_ids = pos_ids + [0]*(11-len(pos_ids))
                    
                self.match_player_lookup[mid][tid] = pids
                self.match_position_lookup[mid][tid] = pos_ids

        # Pre-index Dataframes
        if 'match_id' in self.team_vectors.columns:
            self.team_vectors = self.team_vectors.set_index(['match_id', 'team_id'])
            
        if not self.momentum.empty and 'match_id' in self.momentum.columns:
            self.momentum = self.momentum.set_index(['match_id', 'team_id'])
            
        if not self.context_history.empty and 'match_id' in self.context_history.columns:
            self.context_history = self.context_history.set_index(['match_id', 'team_id'])

    def _build_h2h_features(self, valid_matches: pd.DataFrame):
        """
        Compute Head-to-Head features for each match using ONLY past meetings.
        
        4-dim feature vector per match:
        - home_h2h_winrate: Home team's win rate in past H2H meetings
        - h2h_draw_rate: Draw rate in past H2H meetings
        - h2h_avg_goals: Average total goals in past H2H meetings
        - h2h_familiarity: log(1 + num_meetings) / log(max_meetings) — how familiar are these teams?
        
        CRITICAL: Temporal integrity — only meetings BEFORE the current match date are used.
        Bob would have our heads if we leaked future H2H data.
        """
        df = valid_matches.sort_values('date').reset_index(drop=True)
        h2h_features = np.zeros((len(df), 4), dtype=np.float32)
        
        # Running history: pair_key → list of (outcome_for_first_team, total_goals)
        pair_history = {}
        
        for i, row in df.iterrows():
            hid = int(row['home_team_id'])
            aid = int(row['away_team_id'])
            pair_key = tuple(sorted([hid, aid]))
            is_first = (hid == pair_key[0])  # Is home team the "first" in sorted pair?
            
            # Get PAST history for this pair
            history = pair_history.get(pair_key, [])
            
            if len(history) >= 1:
                outcomes = [h[0] for h in history]
                goals = [h[1] for h in history]
                n = len(history)
                
                # Win rate from home team's perspective
                if is_first:
                    wins = sum(1 for o in outcomes if o == 0)  # 0=first team won
                else:
                    wins = sum(1 for o in outcomes if o == 2)  # 2=second team won
                draws = sum(1 for o in outcomes if o == 1)
                
                h2h_features[i, 0] = wins / n           # home_h2h_winrate
                h2h_features[i, 1] = draws / n           # h2h_draw_rate
                h2h_features[i, 2] = np.mean(goals) / 5.0  # h2h_avg_goals (normalized by 5)
                h2h_features[i, 3] = np.log1p(n) / np.log1p(10)  # familiarity (capped at 10)
            # else: all zeros (no history)
            
            # Now ADD this match to history (for future matches to use)
            if not self.inference_mode:
                outcome = int(row['outcome'])
                total_goals = float(row['home_score']) + float(row['away_score'])
                # Store outcome relative to "first" team in sorted pair
                if is_first:
                    stored_outcome = outcome  # 0=home(first) won, 1=draw, 2=away(second) won
                else:
                    stored_outcome = 2 - outcome if outcome != 1 else 1  # Flip perspective
                pair_history.setdefault(pair_key, []).append((stored_outcome, total_goals))
        
        self.h2h_features = h2h_features
        n_with_history = (h2h_features[:, 3] > 0).sum()
        logger.info(f"H2H features built: {n_with_history}/{len(df)} matches have past H2H data")

    def __len__(self):
        return len(self.match_ids)
    
    def __getitem__(self, idx):
        mid = self.match_ids[idx]
        hid = self.home_ids[idx]
        aid = self.away_ids[idx]
        
        # 1. Player Indices & Grid Position
        h_players = self.match_player_lookup.get(mid, {}).get(hid, [0]*11)
        a_players = self.match_player_lookup.get(mid, {}).get(aid, [0]*11)
        
        h_positions = self.match_position_lookup.get(mid, {}).get(hid, [0]*11)
        a_positions = self.match_position_lookup.get(mid, {}).get(aid, [0]*11)
        
        h_man = self.match_home_man_lookup.get(mid, 0)
        a_man = self.match_away_man_lookup.get(mid, 0)
        
        # 2. Team Context (XI Rating + History + Home Flag)
        # 11 XI features + 5 History features + 1 Home Flag = 17 total
        
        def get_full_context(tid, is_home):
            # XI Stats
            try:
                xi = self.team_vectors.loc[(mid, tid)].values.astype(np.float32)
            except KeyError:
                xi = np.zeros(11, dtype=np.float32)
                
            # History Stats
            try:
                if self.context_history.empty: hist = np.zeros(5, dtype=np.float32)
                else: hist = self.context_history.loc[(mid, tid)].values.astype(np.float32)
            except KeyError:
                 hist = np.zeros(5, dtype=np.float32)
            
            # Home Advantage Flag
            home_flag = np.array([1.0 if is_home else 0.0], dtype=np.float32)
                 
            return np.concatenate([xi, hist, home_flag])
                
        h_ctx = get_full_context(hid, is_home=True)
        a_ctx = get_full_context(aid, is_home=False)
        
        # 3. Momentum
        def get_mom_vec(tid):
            try:
                if self.momentum.empty: return np.zeros(13, dtype=np.float32)
                return self.momentum.loc[(mid, tid)].values.astype(np.float32)
            except KeyError:
                return np.zeros(self.momentum.shape[1], dtype=np.float32)
                
        h_mom = get_mom_vec(hid)
        a_mom = get_mom_vec(aid)
        
        item = {
            'home_players': torch.LongTensor(h_players),
            'away_players': torch.LongTensor(a_players),
            'home_positions': torch.LongTensor(h_positions),
            'away_positions': torch.LongTensor(a_positions),
            'home_manager': torch.tensor(h_man, dtype=torch.long),
            'away_manager': torch.tensor(a_man, dtype=torch.long),
            'home_context': torch.FloatTensor(h_ctx),
            'away_context': torch.FloatTensor(a_ctx),
            'home_momentum': torch.FloatTensor(h_mom),
            'away_momentum': torch.FloatTensor(a_mom),
            'league_id': torch.tensor(self.league_ids[idx], dtype=torch.long),
            'venue_id': torch.tensor(self.venue_ids[idx], dtype=torch.long),
            'season_phase': torch.tensor(self.season_phases[idx], dtype=torch.long),
            'h2h_features': torch.FloatTensor(self.h2h_features[idx]),
            'referee_id': torch.tensor(self.match_ref_lookup.get(mid, 0), dtype=torch.long),
        }
        
        if not self.inference_mode:
            item['outcome'] = torch.tensor(self.outcomes[idx], dtype=torch.long)
            item['home_goals'] = torch.tensor(self.home_goals[idx], dtype=torch.float)
            item['away_goals'] = torch.tensor(self.away_goals[idx], dtype=torch.float)
            
        return item
