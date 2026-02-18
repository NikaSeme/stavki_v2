
import time
import numpy as np
import pandas as pd
from typing import Dict
from stavki.strategy.kelly import KellyStaker

def create_dummy_bets(n=10000):
    bets = []
    print(f"Generating {n} dummy bets...")
    for i in range(n):
        prob = np.random.uniform(0.4, 0.8)
        odds = 1.0 / (prob - 0.05) # Positive EV on average
        result = "win" if np.random.random() < prob else "loss"
        
        bets.append({
            "model_prob": prob,
            "odds": odds,
            "result": result,
            "status": result
        })
    return bets

def benchmark_kelly():
    bets = create_dummy_bets(10000)
    
    print("Benchmarking KellyStaker (vectorized)...")
    staker = KellyStaker()
    
    start_time = time.time()
    
    # Run optimization
    best_f, results = staker.optimize_kelly_fraction(bets)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Optimized {len(bets)} bets over 10 fractions in {duration:.4f} seconds")
    print(f"Best fraction: {best_f}")
    print(f"Best ROI: {results[best_f]['roi']:.2%}")

if __name__ == "__main__":
    benchmark_kelly()
