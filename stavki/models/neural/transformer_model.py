
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from ..base import BaseModel, Prediction, Market
import logging

logger = logging.getLogger(__name__)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, output_dim: int):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) # Max seq len 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(d_model * 2, output_dim) # Home + Away
        
    def forward(self, home_seq, away_seq):
        # home_seq: (Batch, SeqLen, Feats)
        b, s, f = home_seq.shape
        
        # Embed
        h = self.embedding(home_seq) + self.pos_encoder[:, :s, :]
        a = self.embedding(away_seq) + self.pos_encoder[:, :s, :]
        
        # Transform
        h_out = self.transformer(h)
        a_out = self.transformer(a)
        
        # Pool (Last item? Mean?)
        # Use last item as it represents "current state" after sequence
        h_state = h_out[:, -1, :]
        a_state = a_out[:, -1, :]
        
        # Combine
        combined = torch.cat([h_state, a_state], dim=1)
        return self.fc(combined)

class TransformerModel(BaseModel):
    """
    V3 Transformer: Sequence-based prediction.
    Takes raw match history (last N matches) for Home and Away teams.
    """
    
    def __init__(self, seq_len: int = 10, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__(name="V3_Transformer", markets=[Market.MATCH_WINNER])
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.model = None
        
    
    def _build_sequences(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build sequences for Home and Away teams for each match.
        Returns: (HomeSeqs, AwaySeqs, Targets)
        """
        # 1. Sort by Date
        df = data.sort_values("Date").copy().reset_index(drop=True)
        
        # 2. Select Raw Features
        # Using basics available in features_full.csv
        # FTHG, FTAG are standard. 
        # Elo might be 'elo_home', 'elo_away'
        # Normalize!
        
        # We need to restructure data to be "Team-Centric"
        # Each match implies TWO records: one for Home, one for Away
        
        # Feature columns for the SEQUENCE (History)
        # We want: [GoalsFor, GoalsAgainst, OpponentElo, IsHome, Result(Win/Draw/Loss)]
        # Result: Win=1, Draw=0, Loss=-1 (or one-hot)
        
        cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "elo_home", "elo_away"]
        # Filter for existing only
        cols = [c for c in cols if c in df.columns]
        
        matches = df[cols].copy()
        
        # Create Team-Centric view
        home_side = matches.rename(columns={
            "HomeTeam": "Team", "AwayTeam": "Opponent", 
            "FTHG": "GF", "FTAG": "GA", 
            "elo_home": "Elo", "elo_away": "OppElo"
        })
        home_side["IsHome"] = 1.0
        
        away_side = matches.rename(columns={
            "AwayTeam": "Team", "HomeTeam": "Opponent", 
            "FTAG": "GF", "FTHG": "GA", 
            "elo_away": "Elo", "elo_home": "OppElo"
        })
        away_side["IsHome"] = 0.0
        
        # Concatenate and sort
        team_matches = pd.concat([home_side, away_side], ignore_index=True)
        team_matches = team_matches.sort_values(["Team", "Date"])
        
        # Calculate Result
        # Win (1), Draw (0.5), Loss (0) ? or 1, 0, -1
        # Let's use normalized inputs 0..1 often better for NN
        # Win=1, Draw=0.5, Loss=0
        # GF, GA normalized? maybe / 5.0
        # Elo normalized? (Elo-1500)/400
        
        team_matches["Result"] = np.where(team_matches["GF"] > team_matches["GA"], 1.0,
                                np.where(team_matches["GF"] == team_matches["GA"], 0.5, 0.0))
        
        team_matches["GF_norm"] = team_matches["GF"] / 5.0
        team_matches["GA_norm"] = team_matches["GA"] / 5.0
        team_matches["Elo_norm"] = (team_matches["Elo"] - 1500) / 400.0
        team_matches["OppElo_norm"] = (team_matches["OppElo"] - 1500) / 400.0
        
        feature_cols = ["GF_norm", "GA_norm", "Elo_norm", "OppElo_norm", "IsHome", "Result"]
        n_features = len(feature_cols)
        
        # Convert to numpy for speed
        # Group by Team
        
        # Map (Team, Date) -> Sequence Index
        # We need to map back to the original 'df' (matches)
        
        # Pre-compute dictionary of team -> array of features
        team_history = {}
        for team, group in team_matches.groupby("Team"):
            # Sort by date
            group = group.sort_values("Date")
            feats = group[feature_cols].values.astype(np.float32)
            team_history[team] = feats
            # Also store Dates to align?
            # Or map Date -> Index in feats
            # group["Date"] is datetime
            # map: date -> idx
            date_map = {d: i for i, d in enumerate(group["Date"])}
            team_history[team + "_dates"] = date_map
            
        # 3. Build Sequences for Main DF
        home_seqs = []
        away_seqs = []
        targets = []
        
        pad_vec = np.zeros(n_features, dtype=np.float32)
        
        for idx, row in df.iterrows():
            date = row["Date"]
            h_team = row["HomeTeam"]
            a_team = row["AwayTeam"]
            
            # Label
            if row["FTHG"] > row["FTAG"]:
                y = 0 # Home
            elif row["FTHG"] == row["FTAG"]:
                y = 1 # Draw
            else:
                y = 2 # Away
            
            # Get histories
            # Home
            if h_team in team_history:
                feats = team_history[h_team]
                date_map = team_history[h_team + "_dates"]
                # Find index of current match
                # Current match IS in the history (we built it from df)
                # We need features BEFORE current match
                if date in date_map:
                    curr_idx = date_map[date]
                    # Slice [curr_idx - seq_len : curr_idx]
                    start = curr_idx - self.seq_len
                    end = curr_idx
                    
                    if start < 0:
                        # Padding needed
                        part = feats[0:end]
                        pad_len = self.seq_len - len(part)
                        seq = np.vstack([np.tile(pad_vec, (pad_len, 1)), part])
                    else:
                        seq = feats[start:end]
                else:
                    # Weird, date mismatch?
                    seq = np.tile(pad_vec, (self.seq_len, 1))
            else:
                 seq = np.tile(pad_vec, (self.seq_len, 1))
            home_seqs.append(seq)
            
            # Away
            if a_team in team_history:
                feats = team_history[a_team]
                date_map = team_history[a_team + "_dates"]
                if date in date_map:
                    curr_idx = date_map[date]
                    start = curr_idx - self.seq_len
                    end = curr_idx
                    
                    if start < 0:
                        part = feats[0:end]
                        pad_len = self.seq_len - len(part)
                        seq = np.vstack([np.tile(pad_vec, (pad_len, 1)), part])
                    else:
                        seq = feats[start:end]
                else:
                    seq = np.tile(pad_vec, (self.seq_len, 1))
            else:
                 seq = np.tile(pad_vec, (self.seq_len, 1))
            away_seqs.append(seq)
            
            targets.append(y)
            
        return (
            torch.tensor(np.array(home_seqs), dtype=torch.float32), 
            torch.tensor(np.array(away_seqs), dtype=torch.float32),
            torch.tensor(np.array(targets), dtype=torch.long)
        )

    def fit(self, data: pd.DataFrame, epochs: int = 10, batch_size: int = 64, lr: float = 0.001, **kwargs) -> Dict[str, float]:
        """
        Train the Transformer model.
        """
        logger.info(f"Fitting V3 Transformer on {len(data)} matches...")
        
        # 1. Build Sequences
        home_seqs, away_seqs, targets = self._build_sequences(data)
        
        # 2. Split (80/20 temporal)
        split_idx = int(len(home_seqs) * 0.8)
        
        train_h = home_seqs[:split_idx]
        train_a = away_seqs[:split_idx]
        train_y = targets[:split_idx]
        
        val_h = home_seqs[split_idx:]
        val_a = away_seqs[split_idx:]
        val_y = targets[split_idx:]
        
        # 3. Init Model
        input_dim = home_seqs.shape[2] # Feature dim
        output_dim = 3 # 1X2
        
        self.model = TimeSeriesTransformer(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.n_heads,
            num_layers=self.n_layers,
            output_dim=output_dim
        )
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # 4. Training Loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            # Batching (simple full batch for now if small, but better batched)
            # permutation
            indices = torch.randperm(len(train_h))
            
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(train_h), batch_size):
                idx = indices[i:i+batch_size]
                bh = train_h[idx]
                ba = train_a[idx]
                by = train_y[idx]
                
                optimizer.zero_grad()
                out = self.model(bh, ba)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(val_h, val_a)
                val_loss = criterion(val_out, val_y).item()
                
                # Accuracy
                preds = torch.argmax(val_out, dim=1)
                acc = (preds == val_y).float().mean().item()
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f} - Acc: {acc:.4%}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save state dict if needed, but we keep in memory
        
        self.is_fitted = True
        return {"val_loss": best_val_loss, "val_acc": acc}

    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        if not self.is_fitted:
            logger.warning("V3 Transformer not fitted!")
            return []
            
        self.model.eval()
        
        # Build sequences (expensive, assumes data contains history or is aligned)
        # For meaningful prediction, we assume 'data' contains the matches we want to predict
        # AND we have history access. 
        # _build_sequences builds history from 'data' itself.
        # This implies 'data' MUST contain past matches too? 
        # Ideally we should pass 'data' + 'history_context'.
        # For this PoC, we assume we are predicting on a dataset that includes history (like validation set).
        
        home_seqs, away_seqs, _ = self._build_sequences(data)
        
        with torch.no_grad():
            logits = self.model(home_seqs, away_seqs)
            probs = torch.softmax(logits, dim=1).numpy()
            
        predictions = []
        # Create match IDs
        from stavki.utils import generate_match_id
        
        # sort logic in _build_sequences means predictions are sorted by date.
        # We must align. _build_sequences returns sorted sequences.
        data_sorted = data.sort_values("Date").reset_index(drop=True)
        
        for i, row in data_sorted.iterrows():
            match_id = row.get("match_id", generate_match_id(row.get("HomeTeam"), row.get("AwayTeam"), row.get("Date")))
            
            p = probs[i]
            # p is [Home, Draw, Away] because targets were 0=H, 1=D, 2=A
            
            predictions.append(Prediction(
                match_id=str(match_id),
                market=Market.MATCH_WINNER,
                probabilities={"home": float(p[0]), "draw": float(p[1]), "away": float(p[2])},
                confidence=float(np.max(p) - np.sort(p)[-2]),
                model_name=self.name
            ))
            
            
        return predictions

    def _get_state(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict() if self.model else None,
            "params": {
                "seq_len": self.seq_len,
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "n_layers": self.n_layers
            }
        }

    def _set_state(self, state: Dict[str, Any]):
        params = state["params"]
        self.seq_len = params["seq_len"]
        self.d_model = params["d_model"]
        self.n_heads = params["n_heads"]
        self.n_layers = params["n_layers"]
        
        # Re-init model structure
        self.model = TimeSeriesTransformer(
            input_dim=6, # Hardcoded for now based on _build_sequences
            d_model=self.d_model,
            nhead=self.n_heads,
            num_layers=self.n_layers,
            output_dim=3
        )
        
        if state["model_state_dict"]:
            self.model.load_state_dict(state["model_state_dict"])
            self.is_fitted = True
