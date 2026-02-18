"""
Goals Regressor - Neural Network for Lambda Prediction
========================================================

Predicts expected goals (λ_home, λ_away) directly.
Used for all goals-based markets (O/U, Asian Handicap, Correct Score).
"""

import numpy as np
import pandas as pd
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

from scipy.stats import poisson

from ..base import BaseModel, Prediction, Market

logger = logging.getLogger(__name__)


class GoalsNetwork(nn.Module):
    """Network that outputs λ_home and λ_away."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Separate heads for λ_home and λ_away
        self.head_home = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensures positive output
        )
        
        self.head_away = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(self, x):
        h = self.encoder(x)
        lambda_home = self.head_home(h).squeeze(-1) + 0.3  # Minimum 0.3 goals expected
        lambda_away = self.head_away(h).squeeze(-1) + 0.3
        return lambda_home, lambda_away


class GoalsRegressor(BaseModel):
    """
    Neural regressor for λ (expected goals) prediction.
    
    Can derive probabilities for:
    - Over/Under (any line)
    - Asian Handicap
    - BTTS
    - Correct Score
    """
    
    SUPPORTED_MARKETS = [
        Market.OVER_UNDER,
        Market.ASIAN_HANDICAP,
        Market.BTTS,
        Market.CORRECT_SCORE,
    ]
    
    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        n_epochs: int = 100,
        batch_size: int = 64,
        features: Optional[List[str]] = None,
    ):
        super().__init__(
            name="GoalsRegressor",
            markets=self.SUPPORTED_MARKETS
        )
        
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        self.features = features or [
            "elo_home", "elo_away", "elo_diff",
            "form_home_gf", "form_home_pts", 
            "form_away_gf", "form_away_pts",
            "synth_xg_home", "synth_xg_away",
            "avg_rating_home", "avg_rating_away",
            "rating_delta",
        ]
        
        self.network: Optional[GoalsNetwork] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")
    
    def fit(self, data: pd.DataFrame, accumulation_steps: int = 1, num_workers: int = 0, pin_memory: bool = False, **kwargs) -> Dict[str, float]:
        """Train goals regressor."""
        df = data.copy()
        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
        
        # Get features
        available = [f for f in self.features if f in df.columns]
        
        if len(available) < 3:
            # Fallback
            available = [c for c in df.columns 
                        if df[c].dtype in [np.float64, np.int64]
                        and c not in ["FTHG", "FTAG"]][:15]
        
        # Targets
        y_home = df["FTHG"].values.astype(float)
        y_away = df["FTAG"].values.astype(float)
        
        # Split
        n = len(df)
        split_idx = int(n * 0.85)
        
        X = df[available].fillna(0).replace([np.inf, -np.inf], 0).values
        
        # Normalize
        self.feature_means = X[:split_idx].mean(axis=0)
        self.feature_stds = X[:split_idx].std(axis=0) + 1e-8
        
        X = (X - self.feature_means) / self.feature_stds
        
        X_train, X_eval = X[:split_idx], X[split_idx:]
        y_h_train, y_h_eval = y_home[:split_idx], y_home[split_idx:]
        y_a_train, y_a_eval = y_away[:split_idx], y_away[split_idx:]
        
        # Network
        self.network = GoalsNetwork(
            input_dim=len(available),
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
        )
        
        # DataLoader
        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_h_train),
            torch.FloatTensor(y_a_train),
        )
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0)
        )
        
        X_eval_t = torch.FloatTensor(X_eval).to(self.device)
        y_h_eval_t = torch.FloatTensor(y_h_eval).to(self.device)
        y_a_eval_t = torch.FloatTensor(y_a_eval).to(self.device)
        
        best_loss = float("inf")
        best_state = None
        patience = 15
        patience_counter = 0
        
        optimizer.zero_grad() 

        for epoch in range(self.n_epochs):
            self.network.train()
            
            for i, batch in enumerate(train_loader):
                x, y_h, y_a = [b.to(self.device) for b in batch]
                
                pred_h, pred_a = self.network(x)
                
                # Poisson NLL loss
                loss_h = F.poisson_nll_loss(pred_h, y_h, log_input=False)
                loss_a = F.poisson_nll_loss(pred_a, y_a, log_input=False)
                loss = loss_h + loss_a
                
                # Gradient Accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Eval
            self.network.eval()
            with torch.no_grad():
                pred_h, pred_a = self.network(X_eval_t)
                eval_loss = (
                    F.poisson_nll_loss(pred_h, y_h_eval_t, log_input=False) +
                    F.poisson_nll_loss(pred_a, y_a_eval_t, log_input=False)
                ).item()
            
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_state = self.network.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        if best_state:
            self.network.load_state_dict(best_state)
        
        # Final metrics
        self.network.eval()
        with torch.no_grad():
            pred_h, pred_a = self.network(X_eval_t)
            mae_home = (pred_h - y_h_eval_t).abs().mean().item()
            mae_away = (pred_a - y_a_eval_t).abs().mean().item()
        
        self.is_fitted = True
        self.metadata["features"] = available
        
        return {
            "mae_home": mae_home,
            "mae_away": mae_away,
            "eval_loss": best_loss,
            "mean_pred_home": pred_h.mean().item(),
            "mean_pred_away": pred_a.mean().item(),
        }
    
    def predict_lambdas(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict λ_home, λ_away for each match as arrays."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        
        features = self.metadata.get("features", [])
        available = [f for f in features if f in data.columns]
        
        X = data[available].fillna(0).values
        # Handle case where self.feature_means is None (should be set in fit)
        if self.feature_means is None:
             self.feature_means = np.zeros(X.shape[1])
             self.feature_stds = np.ones(X.shape[1])

        X = (X - self.feature_means) / self.feature_stds
        X_t = torch.FloatTensor(X).to(self.device)
        
        self.network.eval()
        with torch.no_grad():
            pred_h, pred_a = self.network(X_t)
        
        return pred_h.cpu().numpy(), pred_a.cpu().numpy()
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate predictions for goals-based markets."""
        lh_arr, la_arr = self.predict_lambdas(data)
        
        # Vectorized probability calculations
        # O/U 2.5: P(goals > 2.5) = 1 - CDF(2, lambda_total)
        lambda_total = lh_arr + la_arr
        over_probs = 1 - poisson.cdf(2, lambda_total)
        under_probs = 1 - over_probs
        
        # BTTS: P(h>0) * P(a>0) = (1 - P(0, lh)) * (1 - P(0, la))
        # PMF(0, lambda) = exp(-lambda)
        # So P(>0) = 1 - exp(-lambda)
        btts_yes = (1 - np.exp(-lh_arr)) * (1 - np.exp(-la_arr))
        btts_no = 1 - btts_yes
        
        predictions = []
        
        # Pre-generate match IDs if needed (vectorized if possible, but strict dependency on row values)
        # We'll stick to iteration for object creation but use pre-computed arrays
        
        from stavki.utils import generate_match_id
        
        # Optimization: Generate match IDs in bulk if possible, but iterating is fine for object creation.
        
        matches = data.to_dict('records')
        
        for i, row in enumerate(matches):
            lh, la = float(lh_arr[i]), float(la_arr[i])
            
            match_id = row.get("match_id")
            if not match_id:
                # Fallback
                match_id = generate_match_id(
                     row.get("HomeTeam", "home"), 
                     row.get("AwayTeam", "away"), 
                     row.get("Date")
                )
            
            # O/U 2.5
            p_over = float(over_probs[i])
            p_under = float(under_probs[i])
            
            predictions.append(Prediction(
                match_id=match_id,
                market=Market.OVER_UNDER,
                probabilities={
                    "over_2.5": p_over,
                    "under_2.5": p_under,
                },
                confidence=abs(p_over - 0.5) * 2,
                model_name=self.name,
                features_used={"lambda_home": lh, "lambda_away": la},
            ))
            
            # BTTS
            p_yes = float(btts_yes[i])
            p_no = float(btts_no[i])
            
            predictions.append(Prediction(
                match_id=match_id,
                market=Market.BTTS,
                probabilities={
                    "yes": p_yes,
                    "no": p_no,
                },
                confidence=abs(p_yes - 0.5) * 2,
                model_name=self.name,
            ))
        
        return predictions
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "params": {
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "n_epochs": self.n_epochs,
                "batch_size": self.batch_size,
            },
            "features": self.features,
            "network_state": self.network.state_dict() if self.network else None,
            "network_config": {
                "input_dim": len(self.metadata.get("features", [])),
                "hidden_dim": self.hidden_dim,
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
        
        config = state["network_config"]
        self.network = GoalsNetwork(**config).to(self.device)
        
        if state["network_state"]:
            self.network.load_state_dict(state["network_state"])
