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
NEURAL_FEATURES = [
    # ELO
    "HomeEloBefore", "AwayEloBefore", "EloDiff", "EloExpHome", "EloExpAway",
    
    # Form
    "Home_GF_L5", "Home_GA_L5", "Home_Pts_L5",
    "Away_GF_L5", "Away_GA_L5", "Away_Pts_L5",
    
    # xG
    "xG_Home_L5", "xGA_Home_L5", "xG_Away_L5", "xGA_Away_L5", "xG_Diff",
    
    # Market
    "Sharp_Divergence", "Odds_Volatility", "Market_Consensus",
    "Odds_1X2_Home", "Odds_1X2_Draw", "Odds_1X2_Away",
    
    # H2H
    "H2H_Home_Win_Pct", "H2H_Goals_Avg", "H2H_BTTS_Pct",
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
    Neural network with shared backbone and multiple heads.
    
    Architecture:
    - Input projection → [128 dims]
    - ResNet blocks × 3 → [128 dims]
    - Heads:
        - 1X2: [128 → 32 → 3]
        - O/U: [128 → 32 → 2]
        - BTTS: [128 → 32 → 2]
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_blocks: int = 3,
        dropout: float = 0.2,
        temperature: float = 1.0,
    ):
        super().__init__()
        
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
        
        # 1X2 head (3 classes: home, draw, away)
        self.head_1x2 = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 3),
        )
        
        # Over/Under head (2 classes: over, under)
        self.head_ou = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 2),
        )
        
        # BTTS head (2 classes: yes, no)
        self.head_btts = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(head_dim, 2),
        )
    
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning logits for all markets.
        
        Returns:
            Dict with keys '1x2', 'ou', 'btts' containing logits
        """
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
    
    def predict_proba(self, x) -> Dict[str, torch.Tensor]:
        """Get softmax probabilities for all markets."""
        logits = self.forward(x)
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
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def fit(
        self,
        data: pd.DataFrame,
        eval_ratio: float = 0.15,
        patience: int = 10,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train multi-task model.
        
        Args:
            data: DataFrame with features and targets
            eval_ratio: Validation set fraction
            patience: Early stopping patience
        """
        df = data.copy()
        
        # Sort by date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        # Get features
        available = [f for f in self.features if f in df.columns]
        
        if len(available) < 5:
            # Fallback to all numeric columns
            available = [c for c in df.columns 
                        if df[c].dtype in [np.float64, np.int64]
                        and c not in ["FTHG", "FTAG"]][:30]
        
        logger.info(f"Using {len(available)} features for neural model")
        
        # Create targets
        df["target_1x2"] = self._create_1x2_target(df)
        df["target_ou"] = ((df["FTHG"] + df["FTAG"]) > 2.5).astype(int)
        df["target_btts"] = ((df["FTHG"] > 0) & (df["FTAG"] > 0)).astype(int)
        
        # Drop rows with missing targets
        df = df.dropna(subset=["target_1x2", "target_ou", "target_btts"])
        
        # Split
        n = len(df)
        split_idx = int(n * (1 - eval_ratio))
        
        train_df = df.iloc[:split_idx]
        eval_df = df.iloc[split_idx:]
        
        # Normalize features
        X_train = train_df[available].fillna(0).values
        X_eval = eval_df[available].fillna(0).values
        
        self.feature_means = X_train.mean(axis=0)
        self.feature_stds = X_train.std(axis=0) + 1e-8
        
        X_train = (X_train - self.feature_means) / self.feature_stds
        X_eval = (X_eval - self.feature_means) / self.feature_stds
        
        # Targets
        y_1x2_train = train_df["target_1x2"].values.astype(int)
        y_ou_train = train_df["target_ou"].values.astype(int)
        y_btts_train = train_df["target_btts"].values.astype(int)
        
        y_1x2_eval = eval_df["target_1x2"].values.astype(int)
        y_ou_eval = eval_df["target_ou"].values.astype(int)
        y_btts_eval = eval_df["target_btts"].values.astype(int)
        
        # Create network
        self.network = MultiTaskNetwork(
            input_dim=len(available),
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
            torch.FloatTensor(X_train),
            torch.LongTensor(y_1x2_train),
            torch.LongTensor(y_ou_train),
            torch.LongTensor(y_btts_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        X_eval_t = torch.FloatTensor(X_eval).to(self.device)
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
                x, y1, y2, y3 = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                
                logits = self.network(x)
                
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
                logits = self.network(X_eval_t)
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
        
        # Restore best model
        if best_state:
            self.network.load_state_dict(best_state)
        
        # Compute final metrics
        self.network.eval()
        with torch.no_grad():
            probs = self.network.predict_proba(X_eval_t)
            
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
        available = [f for f in features if f in data.columns]
        
        X = data[available].fillna(0).values
        X = (X - self.feature_means) / self.feature_stds
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.network.eval()
        predictions = []
        
        with torch.no_grad():
            probs = self.network.predict_proba(X_t)
            
            for idx, row in data.iterrows():
                i = data.index.get_loc(idx)
                match_id = row.get("match_id", f"{row.get('HomeTeam', 'home')}_vs_{row.get('AwayTeam', 'away')}")
                
                # 1X2 prediction
                p1x2 = probs["1x2"][i].cpu().numpy()
                predictions.append(Prediction(
                    match_id=match_id,
                    market=Market.MATCH_WINNER,
                    probabilities={
                        "home": float(p1x2[0]),
                        "draw": float(p1x2[1]),
                        "away": float(p1x2[2]),
                    },
                    confidence=float(p1x2.max() - np.partition(p1x2, -2)[-2]),
                    model_name=self.name,
                ))
                
                # O/U prediction
                pou = probs["ou"][i].cpu().numpy()
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
                pbtts = probs["btts"][i].cpu().numpy()
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
    
    def _get_state(self) -> Dict[str, Any]:
        return {
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
            "network_state": self.network.state_dict() if self.network else None,
            "network_config": {
                "input_dim": len(self.metadata.get("features", [])),
                "hidden_dim": self.hidden_dim,
                "n_blocks": self.n_blocks,
                "dropout": self.dropout,
            },
            "feature_means": self.feature_means,
            "feature_stds": self.feature_stds,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        params = state["params"]
        for key, value in params.items():
            setattr(self, key, value)
        
        self.features = state["features"]
        self.feature_means = state["feature_means"]
        self.feature_stds = state["feature_stds"]
        
        # Rebuild network
        config = state["network_config"]
        self.network = MultiTaskNetwork(**config).to(self.device)
        
        if state["network_state"]:
            self.network.load_state_dict(state["network_state"])
