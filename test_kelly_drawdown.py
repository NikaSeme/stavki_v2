import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path("/Users/macuser/Documents/something/stavki_v2")
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.strategy.kelly import KellyStaker

staker = KellyStaker(bankroll=1000.0)

# Simulate history
now = datetime.now()
staker.bet_history = [
    {"settled_at": (now - timedelta(days=10)).isoformat(), "bankroll_after": 2000.0}, # Too old to be peak
    {"settled_at": (now - timedelta(days=5)).isoformat(), "bankroll_after": 1500.0},  # True 7-day peak
    {"settled_at": (now - timedelta(days=1)).isoformat(), "bankroll_after": 1200.0},  # Decreasing
]

staker.bankroll = 1200.0 # Current
# Peak = 1500, Current = 1200 -> Drawdown = 300 / 1500 = 20%
# Pause is 25%, Reduce is 15%. Range is 10%. We are at 20%, which is exactly 50% through the range.
# Therefore multiplier should be 0.5

print(f"Current Drawdown: {staker._get_current_drawdown():.2%}")

# Kelly full = 1.0 (just to test scale)
from stavki.strategy.ev import EVResult
ev = EVResult(match_id="test", market="1X2", selection="H", model_prob=0.6, odds=2.0, ev=0.2, edge_pct=0.2, implied_prob=0.5)
res = staker.calculate_stake(ev_result=ev, apply_limits=True)
print(f"Base Kelly (0.75 fraction of full, full is 0.20): 0.150") # Prob 0.6, odds 2.0 -> (1 * 0.6 - 0.4) / 1 = 0.2
print(f"Expected final pct after 20% drawdown (penalty=50%): 0.150 * 0.5 = 0.075")
print(f"Final Stake Pct: {res.stake_pct:.3f}")
# Expected: 0.150 * 0.5 = 0.075 -> 7.5%
