
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Tuple
from pathlib import Path
from sklearn.metrics import log_loss

from stavki.data.schemas import Match, Team, League, Outcome
from stavki.features.builders.elo import EloCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(path: str) -> List[Match]:
    """Load and parse matches for ELO."""
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date")
    
    matches = []
    for idx, row in df.iterrows():
        try:
            # Map result
            res = row.get("FTR", row.get("Result"))
            if res == "H":
                outcome = Outcome.HOME
            elif res == "A":
                outcome = Outcome.AWAY
            else:
                outcome = Outcome.DRAW
                
            m = Match(
                id=str(idx),
                commence_time=row["Date"],
                home_team=Team(name=row["HomeTeam"]),
                away_team=Team(name=row["AwayTeam"]),
                league=League.EPL, # Placeholder
                home_score=int(row["FTHG"]),
                away_score=int(row["FTAG"]),
                result=outcome,
                is_completed=True
            )
            matches.append(m)
        except Exception:
            continue
            
    return matches

def evaluate_k(matches: List[Match], k: float) -> float:
    """Run ELO with given K and return Log Loss."""
    calc = EloCalculator(k_factor=k, use_dynamic_k=True)
    
    y_true = []
    y_pred = []
    
    # Warmup period? Or evaluate all?
    # Usually evaluate after some history.
    # Let's say first 100 matches are warmup.
    
    warmup = 500
    
    for i, m in enumerate(matches):
        home = m.home_team.normalized_name
        away = m.away_team.normalized_name
        
        # Predict before update
        p_home, p_draw, p_away = calc.predict_match(home, away)
        
        # Store prediction
        if i > warmup:
            if m.result == Outcome.HOME:
                truth = 0
            elif m.result == Outcome.DRAW:
                truth = 1
            else:
                truth = 2
                
            y_true.append(truth)
            y_pred.append([p_home, p_draw, p_away])
            
        # Update
        calc.process_match(match=m)
        
    if not y_true:
        return float('inf')
        
    loss = log_loss(y_true, y_pred, labels=[0, 1, 2])
    return loss

def main():
    data_path = "data/training_data.csv"
    if not Path(data_path).exists():
        print(f"Data file not found: {data_path}")
        return

    print("Loading data...")
    matches = load_data(data_path)
    print(f"Loaded {len(matches)} matches.")
    
    k_grid = [10, 15, 20, 24, 28, 32, 36, 40, 50, 60]
    results = {}
    
    print(f"Grid search K: {k_grid}")
    print("-" * 30)
    print(f"{'K-Factor':<10} | {'Log Loss':<10}")
    print("-" * 30)
    
    best_k = None
    best_loss = float('inf')
    
    for k in k_grid:
        loss = evaluate_k(matches, k)
        results[k] = loss
        print(f"{k:<10} | {loss:.5f}")
        
        if loss < best_loss:
            best_loss = loss
            best_k = k
            
    print("-" * 30)
    print(f"Best K-Factor: {best_k} (Loss: {best_loss:.5f})")

if __name__ == "__main__":
    main()
