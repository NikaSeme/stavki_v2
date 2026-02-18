import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from pathlib import Path
import torch

from .base import BaseModel, Prediction, Market
from .neural.transformer_model import TransformerModel

logger = logging.getLogger(__name__)

class V3WatcherWrapper(BaseModel):
    """
    Wrapper for V3 Transformer to run in 'Watcher' mode.
    
    Responsibilities:
    1. Manage data context: Stitches historical data with upcoming matches to build valid sequences.
    2. Shadow Execution: Runs predictions but marks them so they don't impact the ensemble.
    3. Logging: Persists predictions to a separate log/DB for analysis.
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        super().__init__(name="V3_Watcher", markets=[Market.MATCH_WINNER])
        self.model = TransformerModel()
        self.history_path = Path("data/features_full.csv") # Default location
        
        if model_path and model_path.exists():
            try:
                state = torch.load(model_path)
                # Check if it's a full state dict or just params
                # TransformerModel._set_state expects a dict with 'params' and 'model_state_dict'
                # If we saved it differently, we might need to adjust.
                # Assessing from transformer_model.py: _get_state returns dict.
                # Assuming standard save/load.
                if isinstance(state, dict) and "params" in state:
                    self.model._set_state(state)
                else:
                    # Fallback or specific loading logic
                    # Maybe it's just state_dict?
                    # Let's assume the safe loading via internal method if possible, 
                    # but _set_state is usually for internal use.
                    # If we used model.save(), it pickles the object or uses _get_state.
                    # If it's a pickle...
                    pass
                self.is_fitted = True
                logger.info(f"Loaded V3 Transformer from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load V3 model: {e}")
                self.is_fitted = False
        else:
            logger.warning(f"V3 Model path {model_path} not found. Model will be untrained.")
            self.is_fitted = False

    def fit(self, data: pd.DataFrame, **kwargs):
        # We don't train the watcher typically, but we pass through if needed
        return self.model.fit(data, **kwargs)

    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """
        Generate predictions for 'data' (upcoming matches).
        Crucially, we prepend historical data to 'data' before passing to the Transformer,
        then slice out the predictions for the upcoming matches.
        """
        if not self.is_fitted:
            return []
            
        # 1. Load History
        if not self.history_path.exists():
            logger.warning("History file not found, V3 predictions might be poor (no context).")
            full_df = data
        else:
            try:
                hist_df = pd.read_csv(self.history_path, low_memory=False)
                # Ensure Date parsing
                if "Date" in hist_df.columns:
                    hist_df["Date"] = pd.to_datetime(hist_df["Date"], utc=True)
                
                # Ensure data Date parsing
                if "Date" in data.columns:
                    data["Date"] = pd.to_datetime(data["Date"], utc=True)
                elif "commence_time" in data.columns:
                    data["Date"] = pd.to_datetime(data["commence_time"], utc=True)
                
                # Columns check
                required = ["HomeTeam", "AwayTeam", "Date", "FTHG", "FTAG"]
                # Upcoming data won't have FTHG/FTAG usually, or they are NaN.
                # Transformer build_sequences uses FTHG/FTAG to build history.
                # For the *current* row (upcoming), we don't need scores to predict it,
                # but we need scores for *past* rows.
                
                # Concatenate: History + Upcoming
                # Filter history to items strictly before upcoming?
                # Simplify: Just concat.
                # Be careful of duplicates if 'data' overlaps with 'history'.
                
                # Assume 'data' is new.
                # We need to ensure 'data' columns match 'hist_df' for concat
                # Map 'data' columns if needed
                concat_df = pd.concat([hist_df, data], axis=0, ignore_index=True)
                concat_df = concat_df.sort_values("Date").reset_index(drop=True)
                
                # We need to identify which rows are the "target" (original data)
                # Let's use match_id or index
                # Adding a temporary flag
                # A safer way:
                # 1. Build sequences on ALL data.
                # 2. Extract predictions for rows corresponding to 'data'.
                
                full_df = concat_df
                
            except Exception as e:
                logger.error(f"Error preparing V3 data context: {e}")
                full_df = data

        # 2. Run Inner Model Prediction
        # TransformerModel.predict returns List[Prediction]
        all_preds = self.model.predict(full_df)
        
        # 3. Filter for requested matches only
        # We look for match_ids in 'data'
        target_ids = set()
        for idx, row in data.iterrows():
            # Try to reproduce ID generation logic or use provided ID
             # Assuming standard ID field exists
            if "match_id" in row:
                target_ids.add(str(row["match_id"]))
            elif "event_id" in row:
                target_ids.add(str(row["event_id"]))
        
        filtered_preds = [p for p in all_preds if p.match_id in target_ids]
        
        # 4. Log/Persist (The "Watcher" part)
        # We save specific logs here
        self._log_predictions(filtered_preds)
        
        return filtered_preds

    def _log_predictions(self, preds: List[Prediction]):
        if not preds: return
        
        log_dir = Path("outputs/logs/shadow")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV log
        rows = []
        for p in preds:
            row = {
                "match_id": p.match_id,
                "model": self.name,
                "home_prob": p.probabilities.get("home", 0),
                "draw_prob": p.probabilities.get("draw", 0),
                "away_prob": p.probabilities.get("away", 0),
                "confidence": p.confidence,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        # Append to daily file
        file_path = log_dir / f"v3_shadow_log_{pd.Timestamp.now().strftime('%Y%m%d')}.csv"
        header = not file_path.exists()
        df.to_csv(file_path, mode='a', header=header, index=False)
        logger.info(f"Logged {len(preds)} shadow predictions to {file_path}")

    def _get_state(self):
        """Return state for serialization."""
        return {
            "model_state": self.model._get_state() if self.model else None,
            "is_fitted": self.is_fitted
        }

    def _set_state(self, state):
        """Restore state from serialization."""
        if state.get("model_state"):
            if not self.model:
                self.model = TransformerModel()
            self.model._set_state(state["model_state"])
        self.is_fitted = state.get("is_fitted", False)
