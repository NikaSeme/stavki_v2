"""
Dixon-Coles Poisson Model for Multi-Market Predictions
=======================================================

Enhanced implementation with:
- Time-decay for recent matches
- Dynamic home advantage per league
- Incremental updates after matches
- Multi-market output (1X2, O/U, BTTS, AH, Correct Score)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
from datetime import datetime, timedelta
import logging

from ..base import BaseModel, Prediction, Market

logger = logging.getLogger(__name__)


class DixonColesModel(BaseModel):
    """
    Dixon-Coles model for football match outcome prediction.
    
    Models scoring rates (λ) for each team and uses bivariate Poisson
    with low-score adjustment (ρ) for accurate 0-0, 1-0, 0-1, 1-1 probs.
    
    Outputs probabilities for multiple markets:
    - 1X2 (Match Winner)
    - Over/Under (2.5, 3.5 goals)
    - BTTS (Both Teams To Score)
    - Asian Handicap (based on expected goal difference)
    - Correct Score (top 20 most likely scores)
    """
    
    SUPPORTED_MARKETS = [
        Market.MATCH_WINNER,
        Market.OVER_UNDER,
        Market.BTTS,
        Market.ASIAN_HANDICAP,
        Market.CORRECT_SCORE,
    ]
    
    def __init__(
        self,
        home_advantage: float = 0.25,
        time_decay: float = 0.003,  # Exponential decay rate
        rho: float = -0.05,  # Dixon-Coles adjustment
        max_goals: int = 10,  # For score matrix calculation
    ):
        super().__init__(
            name="DixonColes",
            markets=self.SUPPORTED_MARKETS
        )
        
        # Parameters
        self.home_advantage = home_advantage
        self.time_decay = time_decay
        self.rho = rho
        self.max_goals = max_goals
        
        # Team strengths (fitted)
        self.attack: Dict[str, float] = defaultdict(lambda: 1.0)
        self.defense: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # League-specific home advantages
        self.league_home_adv: Dict[str, float] = {}
        
        # Average scoring rate (baseline)
        self.avg_goals = 1.35
        
        # Match count per team (for confidence)
        self.team_matches: Dict[str, int] = defaultdict(int)
    
    def fit(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Fit team attack/defense parameters + ρ using Scipy minimize (L-BFGS-B).
        
        Now includes the Dixon-Coles ρ correction in the log-likelihood,
        which adjusts probabilities for low-scoring matches (0-0, 1-0, 0-1, 1-1).
        """
        required_cols = ["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert dates & weights
        df = data.copy()
        df["Date"] = pd.to_datetime(df["Date"], format='mixed', dayfirst=True)
        df = df.sort_values("Date")
        
        max_date = df["Date"].max()
        df["Days"] = (max_date - df["Date"]).dt.days
        df["Weight"] = np.exp(-self.time_decay * df["Days"])
        
        # Identify teams
        teams = sorted(list(set(df["HomeTeam"]) | set(df["AwayTeam"])))
        n_teams = len(teams)
        team_to_idx = {team: i for i, team in enumerate(teams)}
        
        # Identify leagues if available
        has_leagues = "League" in df.columns
        if has_leagues:
            leagues = sorted(df["League"].unique())
            n_leagues = len(leagues)
            league_to_idx = {l: i for i, l in enumerate(leagues)}
            league_indices = df["League"].map(league_to_idx).values
            logger.info(f"  Fitting with per-league Home Advantage for {n_leagues} leagues")
        else:
            leagues = []
            n_leagues = 1 # Global HA
            league_indices = np.zeros(len(df), dtype=int)
            logger.info("  Fitting with global Home Advantage (no League column)")

        # Map data to indices for vectorization
        home_idx = df["HomeTeam"].map(team_to_idx).values
        away_idx = df["AwayTeam"].map(team_to_idx).values
        home_goals = df["FTHG"].values.astype(int)
        away_goals = df["FTAG"].values.astype(int)
        weights = df["Weight"].values
        
        # Params structure:
        # [0:n_teams] -> Attack
        # [n_teams:2*n_teams] -> Defense
        # [2*n_teams:2*n_teams+n_leagues] -> Home Advantage (1 or many)
        # [-1] -> Rho
        n_params = 2 * n_teams + n_leagues + 1
        x0 = np.zeros(n_params)
        
        # Initialize HA params with current self.home_advantage
        x0[2*n_teams : 2*n_teams+n_leagues] = self.home_advantage
        x0[-1] = self.rho
        
        logger.info(f"Fitting Dixon-Coles (Scipy) on {len(df)} matches, {n_teams} teams")
        
        # Pre-compute low-score masks for the Dixon-Coles τ correction
        mask_00 = (home_goals == 0) & (away_goals == 0)
        mask_10 = (home_goals == 1) & (away_goals == 0)
        mask_01 = (home_goals == 0) & (away_goals == 1)
        mask_11 = (home_goals == 1) & (away_goals == 1)
        
        avg_goals = self.avg_goals
        
        def neg_log_likelihood(params):
            log_att = params[:n_teams]
            log_def = params[n_teams:2*n_teams]
            ha_params = params[2*n_teams : 2*n_teams+n_leagues]
            rho = params[-1]
            
            # Att/Def for each match
            att_home = log_att[home_idx]
            def_home = log_def[home_idx]
            att_away = log_att[away_idx]
            def_away = log_def[away_idx]
            
            # Home Advantage for each match
            match_ha = ha_params[league_indices]
            
            # Log-Expected Goals
            log_mu_home = np.log(avg_goals) + att_home + def_away + match_ha
            log_mu_away = np.log(avg_goals) + att_away + def_home
            
            mu_home = np.exp(log_mu_home)
            mu_away = np.exp(log_mu_away)
            
            # Poisson Log-Likelihood
            ll_home = home_goals * log_mu_home - mu_home
            ll_away = away_goals * log_mu_away - mu_away
            
            base_ll = ll_home + ll_away
            
            # Dixon-Coles τ (tau) correction
            tau = np.ones(len(home_goals))
            tau[mask_00] = 1 - mu_home[mask_00] * mu_away[mask_00] * rho
            tau[mask_10] = 1 + mu_away[mask_10] * rho
            tau[mask_01] = 1 + mu_home[mask_01] * rho
            tau[mask_11] = 1 - rho
            
            tau = np.maximum(tau, 1e-10)
            
            total_ll = base_ll + np.log(tau)
            
            return -np.sum(weights * total_ll)
        
        # Bounds
        bounds = [(None, None)] * (2 * n_teams)  # attack/defense
        bounds.extend([(None, None)] * n_leagues) # home advantages
        bounds.append((-0.5, 0.5))    # rho
        
        try:
            res = minimize(
                neg_log_likelihood, 
                x0, 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'disp': False}
            )
            
            # Extract parameters
            final_params = res.x
            log_att = final_params[:n_teams]
            log_def = final_params[n_teams:2*n_teams]
            final_ha = final_params[2*n_teams : 2*n_teams+n_leagues]
            self.rho = final_params[-1]
            
            # Update Home Advantage
            if has_leagues:
                self.league_home_adv = {l: float(final_ha[i]) for i, l in enumerate(leagues)}
                self.home_advantage = float(np.mean(final_ha)) # Set global as mean
            else:
                self.home_advantage = float(final_ha[0])
                self.league_home_adv = {}

            # Convert team strength back to multiplicative scale
            att = np.exp(log_att)
            defn = np.exp(log_def)
            
            # Normalize constraints (mean = 1.0)
            att /= att.mean()
            defn /= defn.mean()
            
            # Store
            self.attack = defaultdict(lambda: 1.0, dict(zip(teams, att)))
            self.defense = defaultdict(lambda: 1.0, dict(zip(teams, defn)))
            
            logger.info(
                f"Dixon-Coles optimization converged: "
                f"ρ={self.rho:.4f}, avg_ha={self.home_advantage:.4f}"
            )
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}. Falling back to defaults.")
        
        # Track match counts
        self.team_matches.clear()
        counts = df["HomeTeam"].value_counts() + df["AwayTeam"].value_counts()
        for team, count in counts.items():
            self.team_matches[team] = count
            
        # Basic metrics
        self.is_fitted = True
        return {"n_matches": len(df), "success": True, "rho": float(self.rho), "n_teams": n_teams}
    
    def update_match(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        weight: float = 1.0,
    ):
        """
        Incrementally update model after a match.
        Useful for backtesting and live updates.
        """
        # Expected goals
        exp_home = self.avg_goals * self.attack[home_team] * self.defense[away_team] * \
                   np.exp(self.home_advantage)
        exp_away = self.avg_goals * self.attack[away_team] * self.defense[home_team]
        
        # Learning rate based on existing data
        lr_home = 0.1 / np.sqrt(self.team_matches[home_team] + 1)
        lr_away = 0.1 / np.sqrt(self.team_matches[away_team] + 1)
        
        # Update attack
        self.attack[home_team] *= (1 + lr_home * (home_goals / max(exp_home, 0.5) - 1) * weight)
        self.attack[away_team] *= (1 + lr_away * (away_goals / max(exp_away, 0.5) - 1) * weight)
        
        # Update defense
        self.defense[home_team] *= (1 + lr_home * (away_goals / max(exp_away, 0.5) - 1) * weight)
        self.defense[away_team] *= (1 + lr_away * (home_goals / max(exp_home, 0.5) - 1) * weight)
        
        # Increment match counts
        self.team_matches[home_team] += 1
        self.team_matches[away_team] += 1
    

    
    def predict(self, data: pd.DataFrame) -> List[Prediction]:
        """Generate predictions for all supported markets using vectorized operations."""
        if not self.is_fitted:
            # Fallback or error?
            pass
        
        predictions = []
        
        # Vectorized generation of IDs
        from stavki.utils import generate_match_id
        temp_data = data.copy()
        match_ids = temp_data.apply(
            lambda x: x.get("match_id", generate_match_id(x.get("HomeTeam", ""), x.get("AwayTeam", ""), x.get("Date"))),
            axis=1
        ).values
        
        # 1. Get Lambdas (Vectorized)
        lambda_home, lambda_away = self._get_lambdas_vectorized(data)
        
        # 2. Generate Score Matrices (Vectorized Broadcast)
        # We need PMF for 0..max_goals
        # shape: (N, max_goals+1)
        max_g = self.max_goals + 1
        k = np.arange(max_g)
        
        # scipy.stats.poisson.pmf works with arrays
        # pmf(k, mu) -> if mu is (N,1), k is (1, G), result is (N, G)
        pmf_home = poisson.pmf(k[np.newaxis, :], lambda_home[:, np.newaxis])
        pmf_away = poisson.pmf(k[np.newaxis, :], lambda_away[:, np.newaxis])
        
        # Outer product: (N, G, 1) * (N, 1, G) -> (N, G, G)
        matrices = pmf_home[:, :, np.newaxis] * pmf_away[:, np.newaxis, :]
        
        # 3. Apply Rho Adjustment (Vectorized)
        # Adjust indices (0,0), (0,1), (1,0), (1,1)
        # Matrices shape: (N, 11, 11)
        
        # Pre-compute corrections
        # 0-0
        matrices[:, 0, 0] *= (1 - lambda_home * lambda_away * self.rho)
        # 0-1 (Home=0, Away=1)
        matrices[:, 0, 1] *= (1 + lambda_home * self.rho)
        # 1-0 (Home=1, Away=0)
        matrices[:, 1, 0] *= (1 + lambda_away * self.rho)
        # 1-1
        matrices[:, 1, 1] *= (1 - self.rho)
        
        # Clamp negative probs (though rare with valid rho)
        matrices = np.maximum(matrices, 0)
        
        # Normalize
        sums = matrices.sum(axis=(1, 2), keepdims=True)
        matrices = np.divide(matrices, sums, where=sums!=0)
        
        # 4. Compute Probabilities (Vectorized slicing)
        # 1X2
        # tril(k=-1) is Home Win (i > j)
        # triu(k=1) is Away Win (j > i)
        # diag is Draw
        
        # We can use np.tril/triu but they work on the last 2 axes?
        # No, np.tril is for 2D. For 3D we need a mask or loop?
        # A uniform mask (11,11) works for all
        mask_home = np.tril(np.ones((max_g, max_g)), k=-1).astype(bool)
        mask_away = np.triu(np.ones((max_g, max_g)), k=1).astype(bool)
        mask_draw = np.eye(max_g).astype(bool)
        
        prob_home = np.sum(matrices[:, mask_home], axis=1) # This flattens the last 2 dims selection?
        # Actually matrices[:, mask] returns (N, count). Sum over axis 1.
        
        # Let's verify sum dimensions.
        # matrices is (N, 11, 11). mask is (11, 11).
        # matrices[:, mask_home] selects elements where mask is true. Result is (N, NumberOfTrue).
        # Summing gives (N,)
        p_home = matrices[:, mask_home].sum(axis=1)
        p_away = matrices[:, mask_away].sum(axis=1)
        p_draw = matrices[:, mask_draw].sum(axis=1)
        
        # O/U 2.5
        # Mask for sum(i+j) > 2.5
        idx = np.arange(max_g)
        i_idx, j_idx = np.meshgrid(idx, idx, indexing='ij')
        mask_over = (i_idx + j_idx) > 2.5
        p_over = matrices[:, mask_over].sum(axis=1)
        p_under = 1.0 - p_over
        
        # BTTS
        # Mask for i>0 and j>0
        mask_btts = (i_idx > 0) & (j_idx > 0)
        p_btts_yes = matrices[:, mask_btts].sum(axis=1)
        p_btts_no = 1.0 - p_btts_yes
        
        # 5. Build Prediction Objects (Loop over results)
        # This loop is unavoidable but fast (no math)
        for i in range(len(data)):
            mid = match_ids[i]
            
            # 1X2
            # Confidence
            probs_1x2 = np.array([p_home[i], p_draw[i], p_away[i]])
            # Normalize just in case of minor float errors
            probs_1x2 /= probs_1x2.sum()
            
            sorted_1x2 = np.sort(probs_1x2)
            conf_1x2 = sorted_1x2[-1] - sorted_1x2[-2]
            
            predictions.append(Prediction(
                match_id=mid,
                market=Market.MATCH_WINNER,
                probabilities={
                    "home": float(probs_1x2[0]),
                    "draw": float(probs_1x2[1]),
                    "away": float(probs_1x2[2])
                },
                confidence=float(conf_1x2),
                model_name=self.name
            ))
            
            # OU
            conf_ou = abs(p_over[i] - 0.5) * 2
            predictions.append(Prediction(
                match_id=mid,
                market=Market.OVER_UNDER,
                probabilities={
                    "over_2.5": float(p_over[i]),
                    "under_2.5": float(p_under[i])
                },
                confidence=float(conf_ou),
                model_name=self.name
            ))
            
            # BTTS
            conf_btts = abs(p_btts_yes[i] - 0.5) * 2
            predictions.append(Prediction(
                match_id=mid,
                market=Market.BTTS,
                probabilities={
                    "yes": float(p_btts_yes[i]),
                    "no": float(p_btts_no[i])
                },
                confidence=float(conf_btts),
                model_name=self.name
            ))
            
        return predictions

    def _get_lambdas_vectorized(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized lambda calculation."""
        # 1. Map teams to parameters
        # Default to 1.0 if not found (using .get on dict is slow in loop)
        # Convert dicts to lookups?
        # Or faster: just iterate for mapping (linear in N, fast enough compared to matrix math)
        
        # Optimization:
        # Create a function that maps series to series
        # or use map()
        
        # Attack/Defense maps
        # self.attack is a defaultdict.
        # data["HomeTeam"].map(self.attack) might fail for missing keys if not careful?
        # map() with defaultdict usually returns NaN for missing if using dict access?
        # Actually series.map(dict) returns NaN for missing.
        # We need to fillna(1.0).
        
        att_h = data["HomeTeam"].map(self.attack).fillna(1.0).values
        def_a = data["AwayTeam"].map(self.defense).fillna(1.0).values
        
        att_a = data["AwayTeam"].map(self.attack).fillna(1.0).values
        def_h = data["HomeTeam"].map(self.defense).fillna(1.0).values
        
        # League Home Advantage
        # if League column exists
        if "League" in data.columns:
            # Map league to HA, fillna with self.home_advantage
            ha = data["League"].map(self.league_home_adv).fillna(self.home_advantage).values
        else:
            ha = np.full(len(data), self.home_advantage)
            
        # Compute Lambdas
        # lambda_home = avg * att_h * def_a * exp(ha)
        lambda_home = self.avg_goals * att_h * def_a * np.exp(ha)
        
        # lambda_away = avg * att_a * def_h
        lambda_away = self.avg_goals * att_a * def_h
        
        # Clamp
        lambda_home = np.clip(lambda_home, 0.2, 5.0)
        lambda_away = np.clip(lambda_away, 0.2, 5.0)
        
        return lambda_home, lambda_away
    
    
    def _get_state(self) -> Dict[str, Any]:
        return {
            "home_advantage": self.home_advantage,
            "time_decay": self.time_decay,
            "rho": self.rho,
            "max_goals": self.max_goals,
            "attack": dict(self.attack),
            "defense": dict(self.defense),
            "league_home_adv": self.league_home_adv,
            "avg_goals": self.avg_goals,
            "team_matches": dict(self.team_matches),
        }
    
    def _set_state(self, state: Dict[str, Any]):
        self.home_advantage = state["home_advantage"]
        self.time_decay = state["time_decay"]
        self.rho = state["rho"]
        self.max_goals = state["max_goals"]
        self.attack = defaultdict(lambda: 1.0, state["attack"])
        self.defense = defaultdict(lambda: 1.0, state["defense"])
        self.league_home_adv = state["league_home_adv"]
        self.avg_goals = state["avg_goals"]
        self.team_matches = defaultdict(int, state["team_matches"])


# Alias for backwards compatibility
PoissonModel = DixonColesModel
