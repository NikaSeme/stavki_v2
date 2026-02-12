"""
Disagreement Signal Calculator.

Detects when model predictions strongly disagree, which can indicate:
1. Information asymmetry (one model has better data)
2. Edge cases where ensemble averaging hurts
3. Opportunities for contrarian betting

High disagreement + high confidence = potential opportunity
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


def calculate_disagreement(
    poisson_probs: List[float],
    catboost_probs: List[float],
    neural_probs: List[float]
) -> Dict[str, float]:
    """
    Calculate disagreement metrics between model predictions.
    
    Args:
        poisson_probs: [p_home, p_draw, p_away] from Poisson model
        catboost_probs: [p_home, p_draw, p_away] from CatBoost
        neural_probs: [p_home, p_draw, p_away] from Neural network
        
    Returns:
        Dict with disagreement metrics
    """
    p = np.array(poisson_probs)
    c = np.array(catboost_probs)
    n = np.array(neural_probs)
    
    # Pairwise differences
    pc_diff = np.abs(p - c).mean()
    pn_diff = np.abs(p - n).mean()
    cn_diff = np.abs(c - n).mean()
    
    # Overall disagreement (average pairwise)
    mean_disagreement = (pc_diff + pn_diff + cn_diff) / 3
    
    # Max disagreement on any outcome
    all_probs = np.stack([p, c, n])
    max_disagreement = np.max(all_probs, axis=0) - np.min(all_probs, axis=0)
    
    # Which outcome has most disagreement
    max_outcome_idx = np.argmax(max_disagreement)
    outcome_names = ["home", "draw", "away"]
    
    # Outlier model detection
    # If one model is far from the other two, it might have special info
    ensemble_mean = np.mean(all_probs, axis=0)
    distances = [np.linalg.norm(m - ensemble_mean) for m in all_probs]
    outlier_idx = np.argmax(distances)
    model_names = ["poisson", "catboost", "neural"]
    
    return {
        "disagreement_mean": float(mean_disagreement),
        "disagreement_max": float(np.max(max_disagreement)),
        "disagreement_outcome": outcome_names[max_outcome_idx],
        "outlier_model": model_names[outlier_idx],
        "outlier_distance": float(distances[outlier_idx]),
        "poisson_catboost_diff": float(pc_diff),
        "poisson_neural_diff": float(pn_diff),
        "catboost_neural_diff": float(cn_diff),
    }


def detect_contrarian_opportunity(
    model_probs: Dict[str, List[float]],
    market_probs: List[float],
    threshold: float = 0.15
) -> Optional[Dict]:
    """
    Detect when models agree with each other but disagree with market.
    
    This is a potential contrarian betting opportunity.
    
    Args:
        model_probs: Dict with model name -> [p_home, p_draw, p_away]
        market_probs: Market implied probabilities (no-vig)
        threshold: Minimum difference to flag
        
    Returns:
        Dict with opportunity details if found, else None
    """
    # Calculate ensemble mean
    all_probs = np.array(list(model_probs.values()))
    ensemble = np.mean(all_probs, axis=0)
    market = np.array(market_probs)
    
    # Model agreement (low std = high agreement)
    model_std = np.std(all_probs, axis=0)
    model_agreement = 1 - model_std.mean()  # Higher = more agreement
    
    # Market divergence
    market_divergence = np.abs(ensemble - market)
    
    # Check each outcome
    outcome_names = ["home", "draw", "away"]
    opportunities = []
    
    for i, outcome in enumerate(outcome_names):
        if market_divergence[i] > threshold and model_std[i] < 0.10:
            # Models agree, but differ from market
            opportunities.append({
                "outcome": outcome,
                "model_prob": float(ensemble[i]),
                "market_prob": float(market[i]),
                "divergence": float(market_divergence[i]),
                "model_agreement": float(1 - model_std[i]),
                "type": "model_favors" if ensemble[i] > market[i] else "market_favors"
            })
    
    if opportunities:
        # Return the one with highest divergence
        best = max(opportunities, key=lambda x: x["divergence"])
        return best
    
    return None


def calculate_confidence_score(
    probs: List[float],
    disagreement: float
) -> float:
    """
    Calculate overall confidence score combining prediction certainty and model agreement.
    
    Args:
        probs: Final ensemble probabilities
        disagreement: Mean disagreement score
        
    Returns:
        Confidence score [0, 1]
    """
    # Certainty: how far from uniform (0.33, 0.33, 0.33)
    max_prob = max(probs)
    certainty = (max_prob - 0.33) / 0.67  # Normalize to [0, 1]
    
    # Agreement: inverse of disagreement
    agreement = 1 - min(disagreement * 3, 1.0)  # Scale disagreement
    
    # Combined score
    # Weight certainty more when models agree
    if agreement > 0.7:
        confidence = 0.7 * certainty + 0.3 * agreement
    else:
        confidence = 0.5 * certainty + 0.5 * agreement
    
    return max(0, min(1, confidence))


class DisagreementBuilder:
    """
    Feature builder for disagreement signals.
    
    Used to detect when models disagree, which can inform bet sizing
    or indicate edge opportunities.
    """
    
    def get_features(
        self,
        poisson_probs: List[float],
        catboost_probs: List[float],
        neural_probs: List[float],
        market_probs: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """Get disagreement features."""
        base = calculate_disagreement(poisson_probs, catboost_probs, neural_probs)
        
        # Add ensemble mean
        all_probs = np.stack([poisson_probs, catboost_probs, neural_probs])
        ensemble = np.mean(all_probs, axis=0)
        
        features = {
            "disagreement_score": base["disagreement_mean"],
            "disagreement_max": base["disagreement_max"],
            "is_high_disagreement": 1 if base["disagreement_mean"] > 0.10 else 0,
        }
        
        # Add confidence
        features["confidence_score"] = calculate_confidence_score(
            list(ensemble),
            base["disagreement_mean"]
        )
        
        # Market divergence if available
        if market_probs:
            market = np.array(market_probs)
            market_div = np.abs(ensemble - market).mean()
            features["market_divergence"] = float(market_div)
            features["is_contrarian"] = 1 if market_div > 0.15 else 0
            
            # Check for contrarian opportunity
            opp = detect_contrarian_opportunity(
                {
                    "poisson": poisson_probs,
                    "catboost": catboost_probs,
                    "neural": neural_probs,
                },
                market_probs
            )
            if opp:
                features["contrarian_outcome"] = ["home", "draw", "away"].index(opp["outcome"])
                features["contrarian_strength"] = opp["divergence"]
        
        return features
