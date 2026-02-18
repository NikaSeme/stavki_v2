
import sys
import os
from unittest.mock import MagicMock
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from stavki.strategy.kelly import KellyStaker
from stavki.pipelines.daily import BetCandidate
from stavki.strategy.ev import EVResult # Import EVResult

def verify_bot_logic():
    print("--- Verifying Bot Calculation Logic ---")
    
    # Mock BetCandidate with confidence AND blended_prob
    bet = BetCandidate(
        match_id="test_match",
        home_team="Team A",
        away_team="Team B",
        league="EPL",
        kickoff=datetime.now(),
        market="1x2",
        selection="home",
        odds=2.0,
        bookmaker="TestBook",
        model_prob=0.55,       # Valid win prob
        market_prob=0.50,
        blended_prob=0.54,     # Valid blended prob
        ev=0.08,               # 8% EV
        edge=0.04,
        stake_pct=0.0,
        stake_amount=0.0,
        confidence=0.05,       # OLD ERROR SOURCE (5% "confidence" != 55% prob)
        justified_score=100,
        divergence_level="low"
    )
    
    staker = KellyStaker(bankroll=1000.0)
    
    # 1. Simulate OLD Logic (Incorrect)
    print("\n[Simulation] Old Logic (using confidence):")
    try:
        ev_res_old = EVResult(
            match_id=bet.match_id, market=bet.market, selection=bet.selection,
            model_prob=bet.confidence, # <--- ERROR
            odds=bet.odds, ev=bet.ev, edge_pct=0.0, implied_prob=1/bet.odds
        )
        res_old = staker.calculate_stake(ev_res_old)
        print(f"  Input Prob: {bet.confidence}")
        print(f"  Stake: ${res_old.stake_amount:.2f}")
        if res_old.stake_amount == 0:
            print("  Result: FAIL (Expected, this was the bug)")
    except Exception as e:
        print(f"  Error: {e}")

    # 2. Simulate NEW Logic (Correct)
    print("\n[Simulation] New Logic (using blended_prob):")
    try:
        prob_to_use = getattr(bet, 'blended_prob', bet.model_prob)
        ev_res_new = EVResult(
            match_id=bet.match_id, market=bet.market, selection=bet.selection,
            model_prob=prob_to_use, # <--- FIXED
            odds=bet.odds, ev=bet.ev, edge_pct=0.0, implied_prob=1/bet.odds
        )
        res_new = staker.calculate_stake(ev_res_new)
        print(f"  Input Prob: {prob_to_use}")
        print(f"  Stake: ${res_new.stake_amount:.2f}")
        
        if res_new.stake_amount > 0:
             print("  Result: PASS (Stake is positive)")
        else:
             print("  Result: FAIL (Stake is still zero)")
             
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    verify_bot_logic()
