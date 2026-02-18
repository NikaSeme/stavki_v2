
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stavki.features.builders.elo import EloBuilder
from stavki.data.schemas import Match, Team, League

def create_dummy_matches(n=1000):
    teams = [f"Team_{i}" for i in range(20)]
    matches = []
    base_date = datetime(2023, 1, 1)
    
    print(f"Generating {n} dummy matches...")
    for i in range(n):
        home = teams[np.random.randint(0, 20)]
        away = teams[np.random.randint(0, 20)]
        while home == away:
            away = teams[np.random.randint(0, 20)]
            
        m = Match(
            id=f"m_{i}",
            home_team=Team(name=home),
            away_team=Team(name=away),
            league=League.EPL,
            commence_time=base_date + timedelta(hours=i),
            home_score=np.random.randint(0, 5),
            away_score=np.random.randint(0, 5),
            matchday=1,
            season="2023"
        )
        matches.append(m)
    return matches

def benchmark_elo():
    matches = create_dummy_matches(5000)
    
    print("Benchmarking EloBuilder (vectorized)...")
    start_time = time.time()
    
    builder = EloBuilder()
    # Batch transform (fit + transform in one go)
    df = builder.transform_matches(matches)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed {len(matches)} matches in {duration:.4f} seconds")
    print(f"Rate: {len(matches)/duration:.2f} matches/sec")
    print(f"Feature shape: {df.shape}")

if __name__ == "__main__":
    benchmark_elo()
