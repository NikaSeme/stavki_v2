"""
Ensemble Calibrator
===================
Post-hoc probability calibration for ensemble predictions.
Combines Isotonic Regression and Temperature Scaling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.isotonic import IsotonicRegression
import logging

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..base import Prediction, Market

logger = logging.getLogger(__name__)


class TemperatureScaling:
    """Temperature scaling calibration for neural networks."""
    
    def __init__(self, n_classes: int = 3):
        self.temperature = 1.0
        self.n_classes = n_classes
    
    def fit(self, logits: np.ndarray, labels: np.ndarray, max_iter: int = 50):
        """Fit temperature using NLL minimization."""
        if not HAS_TORCH:
            return
        
        # Find optimal temperature via grid search
        best_temp = 1.0
        best_nll = float("inf")
        
        for temp in np.linspace(0.5, 3.0, 50):
            scaled = logits / temp
            probs = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
            
            # NLL
            nll = -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-10))
            
            if nll < best_nll:
                best_nll = nll
                best_temp = temp
        
        self.temperature = best_temp
        logger.info(f"Optimal temperature: {best_temp:.3f}")
    
    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to probabilities."""
        if self.temperature == 1.0:
            return probs
        
        # Convert to logits
        logits = np.log(np.clip(probs, 1e-10, 1 - 1e-10))
        
        # Scale
        scaled = logits / self.temperature
        
        # Back to probs
        calibrated = np.exp(scaled) / np.exp(scaled).sum(axis=1, keepdims=True)
        
        return calibrated


class EnsembleCalibrator:
    """
    Multi-method calibration for ensemble predictions.
    
    Supports:
    - Isotonic Regression (per outcome)
    - Temperature Scaling (for neural outputs)
    - Platt Scaling (sigmoid calibration)
    """
    
    def __init__(self, method: str = "isotonic"):
        """
        Args:
            method: 'isotonic', 'temperature', or 'platt'
        """
        self.method = method
        self.calibrators: Dict[str, Any] = {}  # market_outcome -> calibrator
        self.is_fitted = False
    
    def fit(
        self, 
        predictions: List[Prediction],
        actuals: Dict[str, Any],  # match_id -> {market_value: outcome} or match_id -> outcome (legacy)
    ):
        """
        Fit calibrators on validation data.
        
        Args:
            predictions: Model predictions
            actuals: Dict mapping match_id to actual outcomes.
                Modern format: {match_id: {"1x2": "home", "btts": "yes", "over_under": "over_2.5"}}
                Legacy format: {match_id: "home"}  (treated as 1X2 only)
        """
        # Normalize actuals to per-market format
        normalized: Dict[str, Dict[str, str]] = {}
        for match_id, val in actuals.items():
            if isinstance(val, dict):
                normalized[match_id] = val
            else:
                # Legacy: assume 1X2
                normalized[match_id] = {"1x2": val}
        
        # Group by market and outcome
        data: Dict[str, Dict[str, List[float]]] = {}  # market_outcome -> {probs, actuals}
        
        for pred in predictions:
            if pred.match_id not in normalized:
                continue
            
            market_key = pred.market.value if hasattr(pred.market, 'value') else str(pred.market)
            match_actuals = normalized[pred.match_id]
            
            # Get the actual outcome for this prediction's market
            actual = match_actuals.get(market_key)
            if actual is None:
                continue
            
            for outcome, prob in pred.probabilities.items():
                key = f"{market_key}_{outcome}"
                
                if key not in data:
                    data[key] = {"probs": [], "actuals": []}
                
                data[key]["probs"].append(prob)
                data[key]["actuals"].append(1 if outcome == actual else 0)
        
        # Fit calibrators
        for key, values in data.items():
            probs = np.array(values["probs"])
            labels = np.array(values["actuals"])
            
            if len(probs) < 20:  # Minimum samples
                logger.info(f"  Skipping {key}: only {len(probs)} samples (need 20)")
                continue
            
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(probs, labels)
                self.calibrators[key] = calibrator
            
            elif self.method == "platt":
                # Sigmoid calibration
                from scipy.optimize import minimize
                
                def sigmoid(x, a, b):
                    return 1 / (1 + np.exp(a * x + b))
                
                def nll(params):
                    a, b = params
                    preds = sigmoid(probs, a, b)
                    return -np.mean(labels * np.log(preds + 1e-10) + 
                                   (1 - labels) * np.log(1 - preds + 1e-10))
                
                result = minimize(nll, [1.0, 0.0], method="L-BFGS-B")
                self.calibrators[key] = ("platt", result.x)
        
        self.is_fitted = True
        logger.info(f"Fitted {len(self.calibrators)} calibrators")
    
    def calibrate(self, predictions: List[Prediction]) -> List[Prediction]:
        """Apply calibration to predictions."""
        if not self.is_fitted:
            return predictions
        
        calibrated = []
        
        for pred in predictions:
            new_probs = {}
            
            for outcome, prob in pred.probabilities.items():
                key = f"{pred.market.value}_{outcome}"
                
                if key in self.calibrators:
                    calibrator = self.calibrators[key]
                    
                    if self.method == "isotonic":
                        new_probs[outcome] = float(calibrator.predict([[prob]])[0])
                    elif self.method == "platt":
                        a, b = calibrator[1]
                        new_probs[outcome] = float(1 / (1 + np.exp(a * prob + b)))
                else:
                    new_probs[outcome] = prob
            
            # Renormalize
            total = sum(new_probs.values())
            if total > 0:
                new_probs = {k: v/total for k, v in new_probs.items()}
            
            calibrated.append(Prediction(
                match_id=pred.match_id,
                market=pred.market,
                probabilities=new_probs,
                confidence=pred.confidence,
                model_name=pred.model_name + "_calibrated",
                features_used=pred.features_used,
            ))
        
        return calibrated
    
    def get_calibration_error(
        self,
        predictions: List[Prediction],
        actuals: Dict[str, Any],  # match_id -> {market_value: outcome} or match_id -> outcome
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute Expected Calibration Error (ECE).
        
        Returns:
            Dict with ECE per market_outcome
        """
        ece = {}
        
        # Normalize actuals to per-market format
        normalized: Dict[str, Dict[str, str]] = {}
        for match_id, val in actuals.items():
            if isinstance(val, dict):
                normalized[match_id] = val
            else:
                normalized[match_id] = {"1x2": val}
        
        # Group by market and outcome
        data: Dict[str, Dict[str, List[float]]] = {}
        
        for pred in predictions:
            if pred.match_id not in normalized:
                continue
            
            market_key = pred.market.value if hasattr(pred.market, 'value') else str(pred.market)
            match_actuals = normalized[pred.match_id]
            actual = match_actuals.get(market_key)
            if actual is None:
                continue
            
            for outcome, prob in pred.probabilities.items():
                key = f"{market_key}_{outcome}"
                
                if key not in data:
                    data[key] = {"probs": [], "actuals": []}
                
                data[key]["probs"].append(prob)
                data[key]["actuals"].append(1 if outcome == actual else 0)
        
        # Compute ECE for each
        for key, values in data.items():
            probs = np.array(values["probs"])
            labels = np.array(values["actuals"])
            
            if len(probs) < n_bins:
                continue
            
            # Bin by predicted probability
            bins = np.linspace(0, 1, n_bins + 1)
            ece_sum = 0.0
            
            for i in range(n_bins):
                mask = (probs >= bins[i]) & (probs < bins[i + 1])
                
                if mask.sum() > 0:
                    bin_probs = probs[mask]
                    bin_labels = labels[mask]
                    
                    avg_pred = bin_probs.mean()
                    avg_actual = bin_labels.mean()
                    
                    ece_sum += mask.sum() * abs(avg_pred - avg_actual)
            
            ece[key] = ece_sum / len(probs)
        
        return ece
