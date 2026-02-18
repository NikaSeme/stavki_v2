
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.strategy.kelly import KellyStaker

class TestKellyStaker:
    
    @pytest.fixture
    def staker(self):
        return KellyStaker(config={"max_stake_pct": 0.05})
    
    def test_kelly_formula(self, staker):
        # p=0.5, b=1.0 (odds=2.0) -> f = (0.5*1 - 0.5)/1 = 0
        assert staker.kelly_formula(0.5, 2.0) == 0.0
        
        # p=0.6, b=1.0 (odds=2.0) -> f = (0.6*1 - 0.4)/1 = 0.2
        assert abs(staker.kelly_formula(0.6, 2.0) - 0.2) < 1e-6
        
        # p=0.5, b=2.0 (odds=3.0) -> f = (0.5*2 - 0.5)/2 = 0.5/2 = 0.25
        assert abs(staker.kelly_formula(0.5, 3.0) - 0.25) < 1e-6

    def test_optimization_vectorized(self, staker):
        # Create predictable bets
        # 10 bets, all win, edge exists
        bets = [
            {"model_prob": 0.6, "odds": 2.0, "result": "win"}
            for _ in range(10)
        ]
        
        # Full Kelly = 0.2
        # Max stake = 0.05
        # If fraction=1.0, stake=0.05. Profit per bet = 0.05 * 1 = 0.05.
        # Total profit = 0.5 (50%).
        
        best_f, results = staker.optimize_kelly_fraction(bets, fractions=[0.5, 1.0])
        
        # Both fractions should be positive
        assert results[1.0]['roi'] > 0
        assert results[0.5]['roi'] > 0
        
    def test_optimization_bankruptcy(self, staker):
        # All loss
        bets = [
            {"model_prob": 0.6, "odds": 2.0, "result": "loss"}
            for _ in range(100)
        ]
        
        best_f, results = staker.optimize_kelly_fraction(bets, fractions=[1.0])
        
        # Should handle it gracefully
        assert results[1.0]['final_bankroll'] < 1000
        assert results[1.0]['max_drawdown'] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
