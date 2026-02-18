"""
Market Adjuster - Meta-Model for Market Signal Integration
============================================================

Adjusts model predictions based on:
- Closing Line Value (CLV)
- Sharp money movement
- Market consensus divergence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..base import BaseModel, Prediction, Market

logger = logging.getLogger(__name__)


class MarketAdjuster(BaseModel):
    """
    Meta-model that adjusts predictions using market signals.
    
    When model prediction diverges from market, and sharp money
    moves toward model, confidence increases. When sharp money
    moves away, we reduce our prediction confidence.
    """
    
    def __init__(
        self,
        clv_weight: float = 0.3,
        sharp_weight: float = 0.4,
        consensus_weight: float = 0.3,
        min_adjustment: float = -0.15,
        max_adjustment: float = 0.15,
    ):
        super().__init__(
            name="MarketAdjuster",
            markets=[Market.MATCH_WINNER, Market.OVER_UNDER, Market.BTTS]
        )
        
        self.clv_weight = clv_weight
        self.sharp_weight = sharp_weight
        self.consensus_weight = consensus_weight
        self.min_adjustment = min_adjustment
        self.max_adjustment = max_adjustment
        
        # Historical accuracy by CLV bucket
        self.clv_accuracy: Dict[str, float] = {}
        
        self.is_fitted = True  # No training needed
    
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Analyze historical CLV vs actual outcomes.
        
        This helps determine optimal adjustment factors.
        """
        if "CLV_Home" not in data.columns:
            return {"note": "No CLV data available"}
        
        # Bucket CLV and compute accuracy
        df = data.copy()
        df["CLV_bucket"] = pd.cut(df["CLV_Home"], bins=[-np.inf, -5, -2, 0, 2, 5, np.inf])
        
        # Compute win rate by CLV bucket
        for bucket, group in df.groupby("CLV_bucket"):
            if len(group) < 20:
                continue
            
            # Assuming model predicted home...
            # This is simplified - real impl needs prediction tracking
            home_wins = (group["FTHG"] > group["FTAG"]).sum()
            self.clv_accuracy[str(bucket)] = home_wins / len(group)
        
        return {"clv_buckets": len(self.clv_accuracy)}
    
    def adjust(
        self,
        predictions: List[Prediction],
        market_data: pd.DataFrame,
    ) -> List[Prediction]:
        """
        Adjust predictions based on market signals.
        
        Args:
            predictions: Original model predictions
            market_data: DataFrame with odds, CLV, sharp signals
            
        Returns:
            Adjusted predictions
        """
        adjusted = []
        
        # Index market data by match_id or HomeTeam/AwayTeam
        market_lookup = {}
        for idx, row in market_data.iterrows():
            key = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}")
            market_lookup[key] = row
        
        for pred in predictions:
            row = market_lookup.get(pred.match_id)
            
            if row is None:
                adjusted.append(pred)
                continue
            
            new_probs = self._adjust_probs(pred, row)
            
            adjusted.append(Prediction(
                match_id=pred.match_id,
                market=pred.market,
                probabilities=new_probs,
                confidence=pred.confidence,
                model_name=pred.model_name + "+adj",
                features_used={
                    **(pred.features_used or {}),
                    "market_adjusted": True,
                },
            ))
        
        return adjusted
    
    def _adjust_probs(
        self,
        pred: Prediction,
        market_row: pd.Series,
    ) -> Dict[str, float]:
        """Apply adjustments to a single prediction."""
        probs = pred.probabilities.copy()
        
        # 1. CLV adjustment
        clv_adj = self._clv_adjustment(pred, market_row)
        
        # 2. Sharp money adjustment
        sharp_adj = self._sharp_adjustment(pred, market_row)
        
        # 3. Consensus adjustment
        consensus_adj = self._consensus_adjustment(pred, market_row)
        
        # Combine adjustments
        total_adj = (
            self.clv_weight * clv_adj +
            self.sharp_weight * sharp_adj +
            self.consensus_weight * consensus_adj
        )
        
        # Clamp total adjustment
        total_adj = np.clip(total_adj, self.min_adjustment, self.max_adjustment)
        
        # Apply adjustment to predicted outcome
        best_outcome = max(probs.items(), key=lambda x: x[1])[0]
        
        # Adjust best outcome probability
        probs[best_outcome] = np.clip(probs[best_outcome] + total_adj, 0.05, 0.95)
        
        # Renormalize
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        return probs
    
    def _clv_adjustment(self, pred: Prediction, row: pd.Series) -> float:
        """
        Positive CLV (we beat closing line) -> increase confidence.
        """
        if pred.market == Market.MATCH_WINNER:
            # Check which outcome we're predicting
            best = max(pred.probabilities.items(), key=lambda x: x[1])[0]
            
            clv_col = {
                "home": "CLV_Home",
                "draw": "CLV_Draw",
                "away": "CLV_Away",
            }.get(best)
            
            if clv_col and clv_col in row:
                clv = row[clv_col]
                # Scale: 5% CLV = +0.05 adjustment
                return float(np.clip(clv / 100, -0.1, 0.1))
        
        return 0.0
    
    def _sharp_adjustment(self, pred: Prediction, row: pd.Series) -> float:
        """
        Sharp money moving toward our prediction -> increase.
        Sharp money moving against -> decrease.
        """
        if "Sharp_Direction" not in row:
            return 0.0
        
        sharp_dir = row["Sharp_Direction"]  # Expected: 'home', 'away', 'draw', or None
        
        if sharp_dir is None:
            return 0.0
        
        best = max(pred.probabilities.items(), key=lambda x: x[1])[0]
        
        if sharp_dir == best:
            return 0.05  # Sharps agree
        else:
            return -0.03  # Sharps disagree (but don't overweight)
    
    def _consensus_adjustment(self, pred: Prediction, row: pd.Series) -> float:
        """
        If our model disagrees with market consensus, adjust based on our
        historical accuracy when disagreeing.
        """
        if "Market_Consensus" not in row:
            return 0.0
        
        consensus = row.get("Market_Consensus")
        best = max(pred.probabilities.items(), key=lambda x: x[1])[0]
        
        if best == consensus:
            return 0.0  # Agreement, no adjustment
        
        # Disagreement - this is where potential edge exists
        if "Sharp_Divergence" in row:
            div = row["Sharp_Divergence"]
            # High divergence + we're on sharp side = more confident
            return float(np.clip(div * 0.02, -0.05, 0.05))
        
        return 0.0
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate market-adjusted predictions."""
        # This model doesn't make predictions on its own
        # It only adjusts other models' predictions
        return []
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "clv_weight": self.clv_weight,
            "sharp_weight": self.sharp_weight,
            "consensus_weight": self.consensus_weight,
            "min_adjustment": self.min_adjustment,
            "max_adjustment": self.max_adjustment,
            "clv_accuracy": self.clv_accuracy,
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.clv_weight = state["clv_weight"]
        self.sharp_weight = state["sharp_weight"]
        self.consensus_weight = state["consensus_weight"]
        self.min_adjustment = state["min_adjustment"]
        self.max_adjustment = state["max_adjustment"]
        self.clv_accuracy = state["clv_accuracy"]
