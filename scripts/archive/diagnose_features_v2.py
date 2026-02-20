
import pandas as pd
from pathlib import Path
from stavki.models.neural.multitask import NEURAL_FEATURES
from stavki.models.neural.goals_regressor import GoalsRegressor

def check_features():
    print("--- Feature Diagnosis ---")
    data_path = "data/features_full.csv"
    if not Path(data_path).exists():
        print("Data file not found!")
        return

    # Read just header
    df = pd.read_csv(data_path, nrows=0)
    csv_cols = set(df.columns)
    
    print(f"CSV Columns ({len(csv_cols)}): {sorted(list(csv_cols))[:10]}...")

    # Check MultiTaskModel
    print("\n[NeuralMultiTask Model]")
    expected = set(NEURAL_FEATURES)
    missing = expected - csv_cols
    present = expected & csv_cols
    print(f"Expected: {len(expected)}")
    print(f"Present:  {len(present)}")
    print(f"Missing:  {len(missing)}")
    if missing:
        print(f"Examples Missing: {list(missing)[:5]}")
        
    # Check GoalsRegressor
    print("\n[GoalsRegressor Model]")
    # Instantiate to get features default
    gr = GoalsRegressor()
    gr_expected = set(gr.features)
    gr_missing = gr_expected - csv_cols
    gr_present = gr_expected & csv_cols
    print(f"Expected: {len(gr_expected)}")
    print(f"Present:  {len(gr_present)}")
    print(f"Missing:  {len(gr_missing)}")
    if gr_missing:
        print(f"Examples Missing: {list(gr_missing)[:5]}")

if __name__ == "__main__":
    check_features()
