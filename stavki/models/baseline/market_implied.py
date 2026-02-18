
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from ..base import BaseModel, Prediction, Market

class MarketImpliedModel(BaseModel):
    """
    Baseline model that uses closing odds to estimate probabilities.
    Removes bookmaker margin using power method or proportional normalization.
    """
    
    def __init__(self, use_margin_removal: bool = True):
        super().__init__(name="MarketImplied", markets=[Market.MATCH_WINNER, Market.OVER_UNDER, Market.BTTS])
        self.use_margin_removal = use_margin_removal
        self.is_fitted = True # No training needed
        
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        return {"status": "skipped_no_training_needed"}
    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        predictions = []
        
        # We need cols like: AvgH, AvgD, AvgA, or B365H...
        # Standardize on 'average' if available, else first found
        
        for idx, row in data.iterrows():
            match_id = row.get("match_id", f"match_{idx}")
            
            # 1X2
            odds_h = row.get("AvgH") or row.get("B365H")
            odds_d = row.get("AvgD") or row.get("B365D")
            odds_a = row.get("AvgA") or row.get("B365A")
            
            if odds_h and odds_d and odds_a:
                probs = np.array([1/odds_h, 1/odds_d, 1/odds_a])
                if self.use_margin_removal:
                    # Basic normalization (Shin is better but slower)
                    probs /= probs.sum()
                
                # Confidence is margin? Or max prob?
                # Let's use max prob - 2nd max prob
                sorted_p = np.sort(probs)
                conf = sorted_p[-1] - sorted_p[-2]
                
                predictions.append(Prediction(
                    match_id=str(match_id),
                    market=Market.MATCH_WINNER,
                    probabilities={"home": probs[0], "draw": probs[1], "away": probs[2]},
                    confidence=float(conf),
                    model_name=self.name
                ))
            
            # O/U 2.5
            o25 = row.get("Avg>2.5") or row.get("B365>2.5")
            u25 = row.get("Avg<2.5") or row.get("B365<2.5")
            
            if o25 and u25:
                probs = np.array([1/o25, 1/u25])
                if self.use_margin_removal:
                    probs /= probs.sum()
                    
                predictions.append(Prediction(
                    match_id=str(match_id),
                    market=Market.OVER_UNDER,
                    probabilities={"over_2.5": probs[0], "under_2.5": probs[1]},
                    confidence=abs(probs[0] - 0.5)*2,
                    model_name=self.name
                ))
                
        return predictions

    def _get_state(self): return {}
    def _set_state(self, state): pass
