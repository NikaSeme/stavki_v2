
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from stavki.strategy.kelly import KellyStaker
from stavki.strategy.ev import EVResult

def test_kelly_scenarios():
    print("--- Testing Kelly Staker Scenarios ---")
    
    # 1. Default Defaults
    staker = KellyStaker(bankroll=1000.0)
    print(f"Config: Min Stake %: {staker.config['min_stake_pct']}, Min Stake $: {staker.config['min_stake_amount']}")
    
    scenarios = [
        # (Odds, Prob, Description)
        (2.0, 0.55, "Strong Value (EV 10%)"),
        (2.0, 0.51, "Marginal Value (EV 2%)"),
        (2.0, 0.501, "Tiny Value (EV 0.2%)"),
        (1.5, 0.70, "Favorite Value (EV 5%)"),
        (5.0, 0.22, "Longshot Value (EV 10%)"),
    ]
    
    for odds, prob, desc in scenarios:
        ev = prob * odds - 1
        ev_res = EVResult(
            match_id="test", market="1x2", selection="home",
            model_prob=prob, odds=odds,
            ev=ev, edge_pct=0.0, implied_prob=1/odds
        )
        
        result = staker.calculate_stake(ev_res)
        
        print(f"\nScenario: {desc}")
        print(f"  Odds: {odds}, Prob: {prob:.4f}, EV: {ev:.4f}")
        print(f"  Full Kelly: {result.kelly_full:.4f}")
        print(f"  Stake Pct: {result.stake_pct:.6f}")
        print(f"  Stake Amt: ${result.stake_amount:.2f}")
        print(f"  Reason: {result.reason}")

if __name__ == "__main__":
    test_kelly_scenarios()
