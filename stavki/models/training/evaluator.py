"""
Model Evaluator - Metrics and ROI Calculation
==============================================

Computes:
- Classification metrics (Accuracy, Log Loss, Brier)
- Calibration error (ECE)
- ROI simulation with Kelly staking
- Per-league breakdown
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
import logging

from ..base import Prediction, Market

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation with betting simulation.
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.03,
        min_prob: float = 0.10,
        max_prob: float = 0.90,
        min_odds: float = 1.2,
        max_odds: float = 10.0,
    ):
        """
        Args:
            kelly_fraction: Fractional Kelly (0.25 = quarter Kelly)
            min_edge: Minimum edge to place bet
            min_prob: Minimum probability threshold
            max_prob: Maximum probability threshold
            min_odds: Minimum acceptable odds
            max_odds: Maximum acceptable odds
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_prob = min_prob
        self.max_prob = max_prob
        self.min_odds = min_odds
        self.max_odds = max_odds
    
    def evaluate(
        self,
        predictions: List[Prediction],
        actuals: pd.DataFrame,
        odds: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation.
        
        Args:
            predictions: Model predictions
            actuals: DataFrame with actual outcomes (FTHG, FTAG)
            odds: Optional odds data for ROI simulation
        
        Returns:
            Dict with all metrics
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "n_predictions": len(predictions),
            "markets": {},
        }
        
        # Group predictions by market
        by_market: Dict[Market, List[Prediction]] = defaultdict(list)
        for pred in predictions:
            by_market[pred.market].append(pred)
        
        # Evaluate each market
        for market, preds in by_market.items():
            market_results = self._evaluate_market(preds, actuals, market)
            results["markets"][market.value] = market_results
        
        # ROI simulation if odds available
        if odds is not None:
            results["roi"] = self._simulate_roi(predictions, actuals, odds)
        
        # Overall metrics
        all_probs = []
        all_actuals = []
        
        for pred in predictions:
            actual = self._get_actual(actuals, pred.match_id, pred.market)
            if actual is None:
                continue
            
            for outcome, prob in pred.probabilities.items():
                all_probs.append(prob)
                all_actuals.append(1 if outcome == actual else 0)
        
        if all_probs:
            results["overall"] = {
                "brier_score": float(np.mean((np.array(all_probs) - np.array(all_actuals))**2)),
                "avg_confidence": float(np.mean([p.confidence for p in predictions])),
            }
        
        return results
    
    def _evaluate_market(
        self,
        predictions: List[Prediction],
        actuals: pd.DataFrame,
        market: Market,
    ) -> Dict[str, float]:
        """Evaluate predictions for a specific market."""
        correct = 0
        total = 0
        log_losses = []
        brier_scores = []
        
        for pred in predictions:
            actual = self._get_actual(actuals, pred.match_id, market)
            if actual is None:
                continue
            
            total += 1
            
            # Best prediction
            best_outcome = max(pred.probabilities.items(), key=lambda x: x[1])[0]
            if best_outcome == actual:
                correct += 1
            
            # Log loss (for actual outcome)
            prob = pred.probabilities.get(actual, 0.01)
            log_losses.append(-np.log(max(prob, 1e-10)))
            
            # Brier score
            for outcome, p in pred.probabilities.items():
                actual_val = 1 if outcome == actual else 0
                brier_scores.append((p - actual_val)**2)
        
        if total == 0:
            return {}
        
        return {
            "accuracy": correct / total,
            "log_loss": float(np.mean(log_losses)),
            "brier_score": float(np.mean(brier_scores)),
            "n_predictions": total,
        }
    
    def _get_actual(
        self,
        actuals: pd.DataFrame,
        match_id: str,
        market: Market,
    ) -> Optional[str]:
        """Get actual outcome for a match."""
        # Try to find by match_id
        for idx, row in actuals.iterrows():
            potential_id = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}_{idx}")
            
            if potential_id == match_id or match_id in str(potential_id):
                if market == Market.MATCH_WINNER:
                    if row["FTHG"] > row["FTAG"]:
                        return "home"
                    elif row["FTHG"] < row["FTAG"]:
                        return "away"
                    else:
                        return "draw"
                
                elif market == Market.OVER_UNDER:
                    total = row["FTHG"] + row["FTAG"]
                    return "over_2.5" if total > 2.5 else "under_2.5"
                
                elif market == Market.BTTS:
                    return "yes" if (row["FTHG"] > 0 and row["FTAG"] > 0) else "no"
        
        return None
    
    def _simulate_roi(
        self,
        predictions: List[Prediction],
        actuals: pd.DataFrame,
        odds: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Simulate ROI with Kelly staking.
        
        Returns:
            Dict with ROI, profit, win rate, etc.
        """
        bankroll = 1000.0  # Starting bankroll
        initial_bankroll = bankroll
        
        bets = []
        
        for pred in predictions:
            if pred.market != Market.MATCH_WINNER:
                continue  # Focus on 1X2 for now
            
            # Find odds
            match_odds = self._get_odds(odds, pred.match_id)
            if not match_odds:
                continue
            
            # Find best value bet
            for outcome, prob in pred.probabilities.items():
                if prob < self.min_prob or prob > self.max_prob:
                    continue
                
                odds_val = match_odds.get(outcome)
                if not odds_val or odds_val < self.min_odds or odds_val > self.max_odds:
                    continue
                
                # Edge = prob - (1/odds)
                implied = 1 / odds_val
                edge = prob - implied
                
                if edge < self.min_edge:
                    continue
                
                # Kelly stake
                kelly = (prob * odds_val - 1) / (odds_val - 1)
                stake_pct = max(0, min(kelly * self.kelly_fraction, 0.05))  # Max 5%
                stake = bankroll * stake_pct
                
                if stake < 1:  # Minimum bet
                    continue
                
                # Get actual outcome
                actual = self._get_actual(actuals, pred.match_id, Market.MATCH_WINNER)
                
                if actual is None:
                    continue
                
                # Resolve bet
                won = (outcome == actual)
                profit = stake * (odds_val - 1) if won else -stake
                bankroll += profit
                
                bets.append({
                    "match_id": pred.match_id,
                    "outcome": outcome,
                    "prob": prob,
                    "odds": odds_val,
                    "edge": edge,
                    "stake": stake,
                    "won": won,
                    "profit": profit,
                })
        
        if not bets:
            return {"error": "No bets placed"}
        
        total_staked = sum(abs(b["stake"]) for b in bets)
        total_profit = bankroll - initial_bankroll
        wins = sum(1 for b in bets if b["won"])
        
        return {
            "total_bets": len(bets),
            "wins": wins,
            "win_rate": wins / len(bets),
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": total_profit / total_staked if total_staked > 0 else 0,
            "final_bankroll": bankroll,
            "avg_edge": np.mean([b["edge"] for b in bets]),
            "avg_odds": np.mean([b["odds"] for b in bets]),
        }
    
    def _get_odds(self, odds: pd.DataFrame, match_id: str) -> Optional[Dict[str, float]]:
        """Get odds for a match."""
        for idx, row in odds.iterrows():
            potential_id = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}_{idx}")
            
            if potential_id == match_id or match_id in str(potential_id):
                return {
                    "home": row.get("Odds_1X2_Home") or row.get("B365H") or row.get("PSH"),
                    "draw": row.get("Odds_1X2_Draw") or row.get("B365D") or row.get("PSD"),
                    "away": row.get("Odds_1X2_Away") or row.get("B365A") or row.get("PSA"),
                }
        
        return None
    
    def calibration_curve(
        self,
        predictions: List[Prediction],
        actuals: pd.DataFrame,
        n_bins: int = 10,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Generate calibration curves for reliability diagram.
        
        Returns:
            Dict mapping market_outcome to list of (mean_pred, mean_actual)
        """
        curves: Dict[str, Dict[str, List]] = defaultdict(lambda: {"preds": [], "actuals": []})
        
        for pred in predictions:
            actual = self._get_actual(actuals, pred.match_id, pred.market)
            if actual is None:
                continue
            
            for outcome, prob in pred.probabilities.items():
                key = f"{pred.market.value}_{outcome}"
                curves[key]["preds"].append(prob)
                curves[key]["actuals"].append(1 if outcome == actual else 0)
        
        result = {}
        
        for key, data in curves.items():
            if len(data["preds"]) < n_bins:
                continue
            
            preds = np.array(data["preds"])
            actuals_arr = np.array(data["actuals"])
            
            # Bin by predicted probability
            bins = np.linspace(0, 1, n_bins + 1)
            curve = []
            
            for i in range(n_bins):
                mask = (preds >= bins[i]) & (preds < bins[i + 1])
                if mask.sum() > 0:
                    curve.append((
                        float(preds[mask].mean()),
                        float(actuals_arr[mask].mean())
                    ))
            
            result[key] = curve
        
        return result
    
    def per_league_breakdown(
        self,
        predictions: List[Prediction],
        actuals: pd.DataFrame,
    ) -> Dict[str, Dict[str, float]]:
        """
        Break down performance by league.
        """
        # Group predictions by league
        by_league: Dict[str, List[Prediction]] = defaultdict(list)
        
        for pred in predictions:
            # Find league from actuals
            for idx, row in actuals.iterrows():
                potential_id = row.get("match_id", f"{row.get('HomeTeam', '')}_{row.get('AwayTeam', '')}_{idx}")
                if potential_id == pred.match_id or pred.match_id in str(potential_id):
                    league = row.get("League", "Unknown")
                    by_league[league].append(pred)
                    break
        
        results = {}
        
        for league, preds in by_league.items():
            league_results = self._evaluate_market(preds, actuals, Market.MATCH_WINNER)
            results[league] = league_results
        
        return results
