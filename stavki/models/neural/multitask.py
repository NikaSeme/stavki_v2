"""
Multi-Task Neural Network for Multi-Market Predictions
=======================================================

PyTorch model with:
- Shared backbone (ResNet blocks)
- Multiple prediction heads (1X2, O/U, BTTS)
- Label smoothing and Focal Loss
- Temperature scaling calibration
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import pickle
import base64

from sklearn.preprocessing import LabelEncoder, StandardScaler

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

from ..base import BaseModel, Prediction, Market

logger = logging.getLogger(__name__)


# Default features for neural model
# Default features for neural model - MUST match features_full.csv
NEURAL_FEATURES = [
    # ELO
    "elo_home", "elo_away", "elo_diff",
    #"EloExpHome", "EloExpAway", # Not in features_full.csv
    
    # Form
    "form_home_gf", "form_home_pts", 
    "form_away_gf", "form_away_pts",
    "form_diff", "gf_diff", "ga_diff",
    
    # xG
    "synth_xg_home", "synth_xg_away", "synth_xg_diff",
    #"xGA_Home_L5", "xGA_Away_L5", # Not in features_full.csv
    
    # Market (Odds & Imp. Prob)
    "B365H", "B365D", "B365A",
    "imp_home_norm", "imp_draw_norm", "imp_away_norm",
    "margin",
    
    # Player / Team Stats
    "avg_rating_home", "avg_rating_away", "rating_delta",
    "key_players_home", "key_players_away",
    "xi_experience_home", "xi_experience_away",
    
    # Referee
    "ref_goals_per_game", "ref_cards_per_game_t1",
    "ref_strictness_t1",
    
    # H2H / Matchup
    "matchup_home_wr", "matchup_sample_size",
    "formation_score_home", "formation_score_away",
    "formation_mismatch",
]


class ResidualBlock(nn.Module):
    """Residual block with GELU activation and layer normalization."""
    
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return x + self.dropout(self.layers(x))


class MultiTaskNetwork(nn.Module):
    """
    Neural network with shared backbone, entity embeddings, and multiple heads.
    
    Architecture:
    - Embeddings (Team/League) → Concatenated with numeric input
    - Input projection → [hidden_dims]
    - ResNet blocks × n
    - Heads: 1X2, O/U, BTTS
    """
    
    def __init__(
        self,
        input_dim: int,
        cat_dims: List[int],     # [num_teams, num_leagues]
        emb_dims: List[int],     # [team_emb_dim, league_emb_dim]
        hidden_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.2,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Embeddings
        # 0: HomeTeam, 1: AwayTeam, 2: League
        self.home_emb = nn.Embedding(cat_dims[0], emb_dims[0])
        self.away_emb = nn.Embedding(cat_dims[0], emb_dims[0])
        self.league_emb = nn.Embedding(cat_dims[1], emb_dims[1])
        
        total_emb_dim = (emb_dims[0] * 2) + emb_dims[1]
        
        # Input projection
        total_input_dim = input_dim + total_emb_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Shared backbone
        self.backbone = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(n_blocks)
        ])
        
        # Prediction heads
        head_dim = 32
        
        self.head_1x2 = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 3),
        )
        
        self.head_ou = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 2),
        )
        
        self.head_btts = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 2),
        )
    
    def forward(self, x_num, x_cat) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_num: Numeric features [batch, n_features]
            x_cat: Categorical features [batch, 3] (Home, Away, League)
        """
        # Embeddings
        home = self.home_emb(x_cat[:, 0])
        away = self.away_emb(x_cat[:, 1])
        league = self.league_emb(x_cat[:, 2])
        
        # Concatenate
        x = torch.cat([x_num, home, away, league], dim=1)
        
        # Shared processing
        h = self.input_proj(x)
        
        for block in self.backbone:
            h = block(h)
        
        # Temperature-scaled logits
        return {
            "1x2": self.head_1x2(h) / self.temperature,
            "ou": self.head_ou(h) / self.temperature,
            "btts": self.head_btts(h) / self.temperature,
        }
    
    def predict_proba(self, x_num, x_cat) -> Dict[str, torch.Tensor]:
        logits = self.forward(x_num, x_cat)
        return {
            "1x2": F.softmax(logits["1x2"], dim=-1),
            "ou": F.softmax(logits["ou"], dim=-1),
            "btts": F.softmax(logits["btts"], dim=-1),
        }


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


class MultiTaskModel(BaseModel):
    """
    Multi-task neural network for multi-market predictions.
    
    Supports:
    - 1X2 (Match Winner)
    - Over/Under 2.5
    - BTTS (Both Teams To Score)
    - Entity Embeddings for Teams and Leagues
    """
    
    SUPPORTED_MARKETS = [Market.MATCH_WINNER, Market.OVER_UNDER, Market.BTTS]
    
    def __init__(
        self,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        n_epochs: int = 50,
        batch_size: int = 64,
        label_smoothing: float = 0.05,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        features: Optional[List[str]] = None,
    ):
        super().__init__(
            name="NeuralMultiTask",
            markets=self.SUPPORTED_MARKETS
        )
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required. Install with: pip install torch")
        
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.features = features or NEURAL_FEATURES
        
        self.network: Optional[MultiTaskNetwork] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Preprocessing
        self.encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
    
    def fit(
        self,
        data: pd.DataFrame,
        eval_ratio: float = 0.15,
        patience: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """Train multi-task model with embeddings."""
        df = data.copy()
        
        # Sort by date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        # Get features
        available = [f for f in self.features if f in df.columns]
        
        if len(available) < 5:
            available = [c for c in df.columns 
                        if df[c].dtype in [np.float64, np.int64]
                        and c not in ["FTHG", "FTAG"]][:30]
        
        logger.info(f"Using {len(available)} features for neural model")
        
        # Prepare categorical data
        # Fill missing with "unknown"
        df["HomeTeam"] = df["HomeTeam"].fillna("unknown")
        df["AwayTeam"] = df["AwayTeam"].fillna("unknown")
        df["League"] = df["League"].fillna("unknown")
        
        # Fit encoders
        le_team = LabelEncoder()
        all_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()
        # Add 'unknown' explicitly if not present
        if "unknown" not in all_teams:
            all_teams = np.append(all_teams, "unknown")
        le_team.fit(all_teams)
        
        le_league = LabelEncoder()
        all_leagues = df["League"].unique()
        if "unknown" not in all_leagues:
            all_leagues = np.append(all_leagues, "unknown")
        le_league.fit(all_leagues)
        
        self.encoders = {
            "team": le_team,
            "league": le_league
        }
        
        # Encode categorical features
        # We process them for the whole DF first to keep logic simple
        # For prediction, we handle unseen labels via 'unknown' mapping
        
        # Note: LabelEncoder transforms to 0..N-1
        # We'll use these directly
        
        # Helper for robust transform
        def safe_transform(le, series):
            # Create a map
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            # Default to 'unknown' index
            unknown_idx = mapping.get("unknown", 0)
            return series.map(mapping).fillna(unknown_idx).astype(int)

        cat_home = safe_transform(le_team, df["HomeTeam"]).values
        cat_away = safe_transform(le_team, df["AwayTeam"]).values
        cat_league = safe_transform(le_league, df["League"]).values
        
        X_cat = np.stack([cat_home, cat_away, cat_league], axis=1) # [N, 3]
        
        # Create targets
        df["target_1x2"] = self._create_1x2_target(df)
        df["target_ou"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        # Drop rows with missing targets (align X_cat as well)
        mask = df[["target_1x2", "target_ou", "target_btts"]].notna().all(axis=1)
        df = df[mask]
        X_cat = X_cat[mask.values]
        
        # Split
        n = len(df)
        split_idx = int(n * (1 - eval_ratio))
        
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]
        
        X_cat_train = X_cat[:split_idx]
        X_cat_eval = X_cat[split_idx:]
        
        # Normalize numeric features
        self.scaler = StandardScaler()
        X_num_train = self.scaler.fit_transform(train_df[available].fillna(0).replace([np.inf, -np.inf], 0).values)
        X_num_eval = self.scaler.transform(eval_df[available].fillna(0).replace([np.inf, -np.inf], 0).values)
        
        # Targets
        y_1x2_train = train_df["target_1x2"].values.astype(int)
        y_ou_train = train_df["target_ou"].values.astype(int)
        y_btts_train = train_df["target_btts"].values.astype(int)
        
        y_1x2_eval = eval_df["target_1x2"].values.astype(int)
        y_ou_eval = eval_df["target_ou"].values.astype(int)
        y_btts_eval = eval_df["target_btts"].values.astype(int)
        
        # Determine embedding dimensions
        n_teams = len(le_team.classes_)
        n_leagues = len(le_league.classes_)
        
        # Rule of thumb: min(50, (N+1)//2)
        dim_team = min(50, (n_teams + 1) // 2)
        dim_league = min(20, (n_leagues + 1) // 2)
        
        logger.info(f"Embeddings: Teams={n_teams} (dim={dim_team}), Leagues={n_leagues} (dim={dim_league})")
        
        # Create network
        self.network = MultiTaskNetwork(
            input_dim=len(available),
            cat_dims=[n_teams, n_leagues],
            emb_dims=[dim_team, dim_league],
            hidden_dim=self.hidden_dim,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=patience // 2, factor=0.5
        )
        
        # Loss functions
        if self.use_focal_loss:
            loss_1x2 = FocalLoss(gamma=self.focal_gamma)
        else:
            loss_1x2 = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        loss_ou = nn.CrossEntropyLoss()
        loss_btts = nn.CrossEntropyLoss()
        
        # DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_num_train),
            torch.LongTensor(X_cat_train),
            torch.LongTensor(y_1x2_train),
            torch.LongTensor(y_ou_train),
            torch.LongTensor(y_btts_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        X_num_eval_t = torch.FloatTensor(X_num_eval).to(self.device)
        X_cat_eval_t = torch.LongTensor(X_cat_eval).to(self.device)
        y_1x2_eval_t = torch.LongTensor(y_1x2_eval).to(self.device)
        y_ou_eval_t = torch.LongTensor(y_ou_eval).to(self.device)
        y_btts_eval_t = torch.LongTensor(y_btts_eval).to(self.device)
        
        # Training loop
        best_loss = float("inf")
        patience_counter = 0
        best_state = None
        
        for epoch in range(self.n_epochs):
            self.network.train()
            train_loss = 0.0
            
            for batch in train_loader:
                x_num, x_cat, y1, y2, y3 = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                
                logits = self.network(x_num, x_cat)
                
                # Combined loss
                l1 = loss_1x2(logits["1x2"], y1)
                l2 = loss_ou(logits["ou"], y2)
                l3 = loss_btts(logits["btts"], y3)
                
                loss = l1 + l2 + l3
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Eval
            self.network.eval()
            with torch.no_grad():
                logits = self.network(X_num_eval_t, X_cat_eval_t)
                eval_l1 = loss_1x2(logits["1x2"], y_1x2_eval_t)
                eval_l2 = loss_ou(logits["ou"], y_ou_eval_t)
                eval_l3 = loss_btts(logits["btts"], y_btts_eval_t)
                eval_loss = (eval_l1 + eval_l2 + eval_l3).item()
            
            scheduler.step(eval_loss)
            
            # Early stopping
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_state = self.network.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Log progress
            if (epoch + 1) % 1 == 0:
                logger.info(f"Epoch {epoch+1}/{self.n_epochs} | Loss: {train_loss:.4f} | Eval: {eval_loss:.4f}")

        
        # Restore best model
        if best_state:
            self.network.load_state_dict(best_state)
        
        # Compute final metrics
        self.network.eval()
        with torch.no_grad():
            probs = self.network.predict_proba(X_num_eval_t, X_cat_eval_t)
            
            acc_1x2 = (probs["1x2"].argmax(dim=1) == y_1x2_eval_t).float().mean().item()
            acc_ou = (probs["ou"].argmax(dim=1) == y_ou_eval_t).float().mean().item()
            acc_btts = (probs["btts"].argmax(dim=1) == y_btts_eval_t).float().mean().item()
        
        self.is_fitted = True
        self.metadata["features"] = available
        self.metadata["n_features"] = len(available)
        
        return {
            "accuracy_1x2": acc_1x2,
            "accuracy_ou": acc_ou,
            "accuracy_btts": acc_btts,
            "best_eval_loss": best_loss,
            "epochs_trained": epoch + 1,
        }
    
    def _create_1x2_target(self, df: pd.DataFrame) -> np.ndarray:
        """Create 1X2 target: 0=Home, 1=Draw, 2=Away."""
        return np.where(
            df["FTHG"] > df["FTAG"], 0,
            np.where(df["FTHG"] < df["FTAG"], 2, 1)
        ).astype(float)
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate predictions for all markets."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        features = self.metadata.get("features", [])
        
        # Align features: ensure all training features exist, fill with 0
        if features:
            # fast alignment using reindex
            X_aligned = data.reindex(columns=features, fill_value=0.0)
            X_aligned = X_aligned.fillna(0.0)
            X_vals = X_aligned.values
        else:
            # Legacy fallback
            available = [f for f in features if f in data.columns]
            X_vals = data[available].fillna(0).values
        
        # Numeric Preprocessing
        try:
            X_num = self.scaler.transform(X_vals)
        except Exception as e:
            logger.error(f"Scaling failed (input shape {X_vals.shape}): {e}")
            return []
            
        X_num_t = torch.FloatTensor(X_num).to(self.device)
        
        # Categorical Preprocessing
        def safe_transform(le, series):
            # Create a map
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            # Default to 'unknown' index
            unknown_idx = mapping.get("unknown", 0)
            return series.map(mapping).fillna(unknown_idx).astype(int)
        
        le_team = self.encoders["team"]
        le_league = self.encoders["league"]
        
        cat_home = safe_transform(le_team, data["HomeTeam"]).values
        cat_away = safe_transform(le_team, data["AwayTeam"]).values
        cat_league = safe_transform(le_league, data["League"]).values
        
        X_cat = np.stack([cat_home, cat_away, cat_league], axis=1) # [N, 3]
        X_cat_t = torch.LongTensor(X_cat).to(self.device)
        
        self.network.eval()
        predictions = []
        
        with torch.no_grad():
            probs = self.network.predict_proba(X_num_t, X_cat_t)
            
            # Convert all to numpy at once
            p1x2_all = probs["1x2"].cpu().numpy()
            pou_all = probs["ou"].cpu().numpy()
            pbtts_all = probs["btts"].cpu().numpy()
            
            matches = data.to_dict('records')
            
            from stavki.utils import generate_match_id
            
            for i, row in enumerate(matches):
                match_id = row.get("match_id")
                if not match_id:
                    match_id = generate_match_id(
                        row.get("HomeTeam", "home"), 
                        row.get("AwayTeam", "away"), 
                        row.get("Date")
                    )
                
                # 1X2 prediction
                p1x2 = p1x2_all[i]
                # Confidence: max - 2nd_max
                # np.partition is fast but for size 3 sorting is trivial
                # sort indices? or just partition
                # For 3 elements, sorting is very fast
                sorted_p = np.sort(p1x2)
                conf_1x2 = float(sorted_p[-1] - sorted_p[-2])
                
                predictions.append(Prediction(
                    match_id=match_id,
                    market=Market.MATCH_WINNER,
                    probabilities={
                        "home": float(p1x2[0]),
                        "draw": float(p1x2[1]),
                        "away": float(p1x2[2]),
                    },
                    confidence=conf_1x2,
                    model_name=self.name,
                ))
                
                # O/U prediction
                pou = pou_all[i]
                predictions.append(Prediction(
                    match_id=match_id,
                    market=Market.OVER_UNDER,
                    probabilities={
                        "over_2.5": float(pou[0]),
                        "under_2.5": float(pou[1]),
                    },
                    confidence=abs(float(pou[0]) - 0.5) * 2,
                    model_name=self.name,
                ))
                
                # BTTS prediction
                pbtts = pbtts_all[i]
                predictions.append(Prediction(
                    match_id=match_id,
                    market=Market.BTTS,
                    probabilities={
                        "yes": float(pbtts[0]),
                        "no": float(pbtts[1]),
                    },
                    confidence=abs(float(pbtts[0]) - 0.5) * 2,
                    model_name=self.name,
                ))
        
        return predictions
    
    def save(self, path: str) -> None:
        """
        Save model to disk using professional split-file strategy.
        
        - weights.pth: PyTorch state dict
        - config.json: Model hyperparameters and metadata
        - preprocessors.joblib: Scikit-learn scalers and encoders
        """
        import json
        import joblib
        import torch
        from pathlib import Path
        
        base_path = Path(path)
        # Create directory if saving to a 'file' path that doesn't have an extension, 
        # or treat 'path' as the base stem.
        # Standard convention: path is a file path like 'models/neural_v1.pt'
        # We will strip extension and use it as a directory or prefix.
        
        # Actually, let's just append suffixes to the given path stem
        parent = base_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        stem = base_path.stem
        
        # 1. Save Weights
        weights_path = parent / f"{stem}_weights.pth"
        if self.network:
            torch.save(self.network.state_dict(), weights_path)
        
        # 2. Save Preprocessors
        preproc_path = parent / f"{stem}_preproc.joblib"
        joblib.dump({
            "encoders": self.encoders,
            "scaler": self.scaler
        }, preproc_path)
        
        # 3. Save Config
        config_path = parent / f"{stem}_config.json"
        
        config = {
            "name": self.name,
            "params": {
                "hidden_dim": self.hidden_dim,
                "n_blocks": self.n_blocks,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
                "label_smoothing": self.label_smoothing,
                "use_focal_loss": self.use_focal_loss,
                "focal_gamma": self.focal_gamma,
            },
            "features": self.features,
            "metadata": self.metadata,
            "network_dims": {
                "input_dim": len(self.metadata.get("features", [])),
                "cat_dims": [
                    len(self.encoders["team"].classes_) if "team" in self.encoders else 0,
                    len(self.encoders["league"].classes_) if "league" in self.encoders else 0,
                ],
                "emb_dims": [
                    self.network.home_emb.embedding_dim if self.network else 0,
                    self.network.league_emb.embedding_dim if self.network else 0,
                ]
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logger.info(f"Saved Neural model to {parent}/{stem}* [.pth, .json, .joblib]")

    @classmethod
    def load(cls, path: str) -> 'MultiTaskModel':
        """Load model from split files."""
        import json
        import joblib
        import torch
        from pathlib import Path
        
        base_path = Path(path)
        parent = base_path.parent
        stem = base_path.stem
        
        # 1. Load Config
        config_path = parent / f"{stem}_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
            
        with open(config_path) as f:
            config = json.load(f)
            
        # Initialize
        instance = cls(**config["params"])
        instance.features = config.get("features", [])
        instance.metadata = config.get("metadata", {})
        instance.is_fitted = True
        
        # 2. Load Preprocessors
        preproc_path = parent / f"{stem}_preproc.joblib"
        if preproc_path.exists():
            preproc = joblib.load(preproc_path)
            instance.encoders = preproc["encoders"]
            instance.scaler = preproc["scaler"]
        
        # 3. Load Weights and Build Network
        weights_path = parent / f"{stem}_weights.pth"
        if weights_path.exists():
            dims = config["network_dims"]
            
            instance.network = MultiTaskNetwork(
                input_dim=dims["input_dim"],
                cat_dims=dims["cat_dims"],
                emb_dims=dims["emb_dims"],
                hidden_dim=config["params"]["hidden_dim"],
                n_blocks=config["params"]["n_blocks"],
                dropout=config["params"]["dropout"]
            ).to(instance.device)
            
            instance.network.load_state_dict(torch.load(weights_path, map_location=instance.device))
            instance.network.eval()
            
        return instance

    def _get_state(self) -> Dict[str, Any]:
        """Legacy state getter - maintained for interface compatibility but discouraged."""
        return {}

    def _set_state(self, state: Dict[str, Any]):
        """Legacy state setter - used when loading old pickle files."""
        if "params" in state:
            params = state["params"]
            for key, value in params.items():
                setattr(self, key, value)
        
        if "features" in state:
            self.features = state["features"]
        
        if "encoders" in state:
            import pickle
            import base64
            try:
                # Attempt to decode if it looks like base64-encoded pickle
                if isinstance(state["encoders"], str):
                    self.encoders = pickle.loads(base64.b64decode(state["encoders"]))
                else:
                    self.encoders = state["encoders"]
                    
                if "scaler" in state:
                    if isinstance(state["scaler"], str):
                        self.scaler = pickle.loads(base64.b64decode(state["scaler"]))
                    else:
                        self.scaler = state["scaler"]
            except Exception as e:
                logger.warning(f"Failed to load encoders/scaler in _set_state: {e}")
        
        # Load network weights if present (and network exists)
        if "network_state" in state and self.network:
            try:
                self.network.load_state_dict(state["network_state"])
            except Exception as e:
                logger.warning(f"Failed to load network state dict: {e}")
        
        # Re-initialize network if config is present and network is missing
        if "network_config" in state and not self.network:
            try:
                config = state["network_config"]
                # Safety check for keys
                if "input_dim" in config:
                    # Initialize network
                    self.network = MultiTaskNetwork(
                        input_dim=config["input_dim"],
                        cat_dims=config["cat_dims"],
                        emb_dims=config["emb_dims"],
                        hidden_dim=config["hidden_dim"],
                        n_blocks=config["n_blocks"],
                        dropout=config["dropout"]
                    ).to(self.device)
                    
                    if "network_state" in state:
                        self.network.load_state_dict(state["network_state"])
                        self.network.eval()
                else:
                    logger.warning("Legacy network config missing required keys 'input_dim'")
                    
            except Exception as e:
                logger.error(f"Failed to rebuild network from state: {e}", exc_info=True)
