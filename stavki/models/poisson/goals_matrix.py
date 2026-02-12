"""
Goals Matrix Utilities
======================
Helper functions for score probability matrices.
"""

import numpy as np
from scipy.stats import poisson
from typing import Dict, List, Tuple, Optional


class GoalsMatrix:
    """
    Utilities for working with score probability matrices.
    
    A score matrix M[i,j] represents P(home_goals=i, away_goals=j).
    """
    
    @staticmethod
    def from_lambdas(
        lambda_home: float,
        lambda_away: float,
        max_goals: int = 10,
        rho: float = 0.0
    ) -> np.ndarray:
        """
        Generate score probability matrix from expected goals.
        
        Args:
            lambda_home: Expected goals for home team
            lambda_away: Expected goals for away team
            max_goals: Maximum goals to consider
            rho: Dixon-Coles low-score adjustment
        
        Returns:
            (max_goals+1, max_goals+1) probability matrix
        """
        n = max_goals + 1
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
                
                # Dixon-Coles adjustment
                if rho != 0:
                    if i == 0 and j == 0:
                        prob *= 1 - lambda_home * lambda_away * rho
                    elif i == 0 and j == 1:
                        prob *= 1 + lambda_home * rho
                    elif i == 1 and j == 0:
                        prob *= 1 + lambda_away * rho
                    elif i == 1 and j == 1:
                        prob *= 1 - rho
                
                matrix[i, j] = max(prob, 0)
        
        # Normalize
        matrix /= matrix.sum()
        return matrix
    
    @staticmethod
    def to_1x2(matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Extract 1X2 probabilities from score matrix.
        
        Returns:
            (home_prob, draw_prob, away_prob)
        """
        home = float(np.triu(matrix, k=1).sum())
        draw = float(np.trace(matrix))
        away = float(np.tril(matrix, k=-1).sum())
        
        total = home + draw + away
        return (home/total, draw/total, away/total)
    
    @staticmethod
    def to_over_under(
        matrix: np.ndarray,
        lines: List[float] = [2.5]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Extract Over/Under probabilities for multiple lines.
        
        Args:
            matrix: Score probability matrix
            lines: Goal lines (e.g., [1.5, 2.5, 3.5])
        
        Returns:
            Dict mapping line to (over_prob, under_prob)
        """
        result = {}
        n = matrix.shape[0]
        
        for line in lines:
            over = 0.0
            under = 0.0
            
            for i in range(n):
                for j in range(n):
                    if i + j > line:
                        over += matrix[i, j]
                    else:
                        under += matrix[i, j]
            
            result[str(line)] = (float(over), float(under))
        
        return result
    
    @staticmethod
    def to_btts(matrix: np.ndarray) -> Tuple[float, float]:
        """
        Extract BTTS (Both Teams To Score) probabilities.
        
        Returns:
            (yes_prob, no_prob)
        """
        yes = float(matrix[1:, 1:].sum())
        no = float(1 - yes)
        return (yes, no)
    
    @staticmethod
    def to_asian_handicap(
        matrix: np.ndarray,
        line: float = 0.0
    ) -> Tuple[float, float, Optional[float]]:
        """
        Calculate Asian Handicap probabilities.
        
        Args:
            matrix: Score probability matrix
            line: Handicap line (positive = home giving goals)
        
        Returns:
            For whole/half lines: (home_cover, away_cover)
            For quarter lines: (home_cover, push, away_cover)
        """
        n = matrix.shape[0]
        home_cover = 0.0
        away_cover = 0.0
        push = 0.0
        
        is_quarter = (line * 4) % 1 == 0 and (line * 2) % 1 != 0
        
        for i in range(n):
            for j in range(n):
                # Goal difference from home perspective
                diff = i - j
                adjusted_diff = diff - line
                
                if is_quarter:
                    # Quarter lines (e.g., -0.25, -0.75)
                    # Split into two half-lines
                    line1 = np.floor(line * 2) / 2
                    line2 = np.ceil(line * 2) / 2
                    
                    # Half on each line
                    prob = matrix[i, j]
                    if diff - line1 > 0:
                        home_cover += prob * 0.5
                    elif diff - line1 < 0:
                        away_cover += prob * 0.5
                    else:
                        push += prob * 0.5
                    
                    if diff - line2 > 0:
                        home_cover += prob * 0.5
                    elif diff - line2 < 0:
                        away_cover += prob * 0.5
                    else:
                        push += prob * 0.5
                else:
                    prob = matrix[i, j]
                    if adjusted_diff > 0:
                        home_cover += prob
                    elif adjusted_diff < 0:
                        away_cover += prob
                    else:
                        push += prob
        
        if (line * 2) % 1 == 0:  # Whole or half line (no push)
            total = home_cover + away_cover + push
            if total > 0:
                return (home_cover/total, away_cover/total, None)
        
        total = home_cover + away_cover + push
        if total > 0:
            return (home_cover/total, push/total, away_cover/total)
        
        return (0.5, 0.0, 0.5)
    
    @staticmethod
    def top_scores(
        matrix: np.ndarray,
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top N most likely correct scores.
        
        Returns:
            List of (score_string, probability) sorted by probability
        """
        scores = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                scores.append((f"{i}-{j}", matrix[i, j]))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:n]
    
    @staticmethod
    def expected_goals(matrix: np.ndarray) -> Tuple[float, float]:
        """
        Calculate expected goals for each team from matrix.
        
        Returns:
            (expected_home_goals, expected_away_goals)
        """
        n = matrix.shape[0]
        exp_home = sum(i * matrix[i, :].sum() for i in range(n))
        exp_away = sum(j * matrix[:, j].sum() for j in range(n))
        return (float(exp_home), float(exp_away))
    
    @staticmethod
    def margin_distributions(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get marginal goal distributions for each team.
        
        Returns:
            (home_distribution, away_distribution) - probability of 0,1,2... goals
        """
        home_dist = matrix.sum(axis=1)
        away_dist = matrix.sum(axis=0)
        return (home_dist, away_dist)
