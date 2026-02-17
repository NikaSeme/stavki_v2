
import time
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from stavki.models.ensemble.predictor import EnsemblePredictor
from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
from stavki.models.poisson.dixon_coles import DixonColesModel
from stavki.backtesting.engine import BacktestEngine
from stavki.config import Config

def create_dummy_data(n_matches=10000):
    print(f"Generating {n_matches} dummy matches...")
    dates = pd.date_range(start="2023-01-01", periods=n_matches, freq="h")
    
    data = pd.DataFrame({
        "Date": dates,
        "HomeTeam": np.random.choice(["Arsenal", "Chelsea", "Liverpool", "Man City", "Spurs"], n_matches),
        "AwayTeam": np.random.choice(["Villa", "Wolves", "Everton", "Newcastle", "Leeds"], n_matches),
        "League": np.random.choice(["E0", "E1", "D1", "I1", "SP1"], n_matches),
        "FTHG": np.random.poisson(1.5, n_matches),
        "FTAG": np.random.poisson(1.2, n_matches),
        "AvgOddsH": np.random.uniform(1.5, 5.0, n_matches),
        "AvgOddsD": np.random.uniform(2.5, 4.0, n_matches),
        "AvgOddsA": np.random.uniform(1.5, 5.0, n_matches),
    })
    
    # Add fake features for LightGBM
    for i in range(50):
        data[f"feature_{i}"] = np.random.randn(n_matches)
        
    return data

def benchmark_models():
    data = create_dummy_data(10000)
    
    print("\n--- Benchmarking Models (10k matches) ---")
    
    # 1. Dixon Coles
    dc = DixonColesModel()
    # Mock fit
    dc.is_fitted = True
    dc.attack = {t: 1.1 for t in data["HomeTeam"].unique()}
    dc.defense = {t: 0.9 for t in data["HomeTeam"].unique()}
    dc.attack.update({t: 1.0 for t in data["AwayTeam"].unique()})
    dc.defense.update({t: 1.0 for t in data["AwayTeam"].unique()})
    
    start = time.time()
    preds_dc = dc.predict(data)
    dc_time = time.time() - start
    print(f"DixonColes: {dc_time:.4f}s ({len(preds_dc)/dc_time:.0f} preds/s)")
    
    # 2. LightGBM
    lgb = LightGBMModel()
    # Mock fit
    lgb.is_fitted = True
    lgb.is_calibrated = False
    lgb.features = [f"feature_{i}" for i in range(50)]
    from unittest.mock import MagicMock
    lgb.model = MagicMock()
    # Mock predict_proba to return random probs
    lgb.model.predict_proba.return_value = np.random.dirichlet(np.ones(3), size=len(data))
    lgb.label_encoder = MagicMock()
    lgb.label_encoder.classes_ = np.array(["A", "D", "H"])
    
    start = time.time()
    preds_lgb = lgb.predict(data)
    lgb_time = time.time() - start
    print(f"LightGBM: {lgb_time:.4f}s ({len(preds_lgb)/lgb_time:.0f} preds/s)")
    
    return preds_dc, preds_lgb

def benchmark_ensemble(data, preds_dc, preds_lgb):
    print("\n--- Benchmarking Ensemble (10k matches) ---")
    
    ensemble = EnsemblePredictor()
    # Mock models inside ensemble to return pre-computed preds
    # Actually ensemble calls predict() on models.
    # We can mock the models list
    
    # But wait, Ensemble.predict calls model.predict(data) internally.
    # So we need to set up the ensemble with the mocked models.
    
    dc = DixonColesModel()
    dc.is_fitted = True
    dc.predict = MagicMock(return_value=preds_dc)
    dc.supports_market = lambda m: True
    
    lgb = LightGBMModel()
    lgb.is_fitted = True
    lgb.predict = MagicMock(return_value=preds_lgb)
    lgb.supports_market = lambda m: True
    
    ensemble.add_model(dc)
    ensemble.add_model(lgb)
    
    start = time.time()
    ens_preds = ensemble.predict(data)
    ens_time = time.time() - start
    print(f"Ensemble: {ens_time:.4f}s ({len(ens_preds)/ens_time:.0f} preds/s)")

def benchmark_backtest(data):
    print("\n--- Benchmarking BacktestEngine (10k matches) ---")
    
    config = Config()
    engine = BacktestEngine(config)
    
    # Mock model_probs
    # dict of match_index -> array
    # But we vectorized engine to use dataframe columns or pre-computed
    # The new engine._generate_signals uses model_probs dict or fallback
    
    # Let's use fallback (market implied) for speed test baseline
    # Or generate a mock dict
    
    start = time.time()
    results = engine.run(data)
    bt_time = time.time() - start
    print(f"Backtest (Implied Probs): {bt_time:.4f}s ({len(data)/bt_time:.0f} matches/s)")

    # 3. Neural MultiTask
    from stavki.models.neural.multitask import MultiTaskModel
    nn_mt = MultiTaskModel()
    nn_mt.is_fitted = True
    # Mock network
    nn_mt.network = MagicMock()
    # Mock probas: dict of tensors or arrays
    # simulate 10k predictions
    n = len(data)
    nn_mt.network.predict_proba.return_value = {
        "1x2": torch.tensor(np.random.dirichlet(np.ones(3), size=n)),
        "ou": torch.tensor(np.random.dirichlet(np.ones(2), size=n)),
        "btts": torch.tensor(np.random.dirichlet(np.ones(2), size=n)),
    }
    nn_mt.encoders = {
        "team": MagicMock(),
        "league": MagicMock(),
    }
    nn_mt.encoders["team"].classes_ = np.array(["A", "B"])
    nn_mt.encoders["league"].classes_ = np.array(["L1", "L2"])
    # Mock scaler
    nn_mt.scaler = MagicMock()
    nn_mt.scaler.transform.return_value = np.random.randn(n, 10)
    
    start = time.time()
    preds_nn = nn_mt.predict(data)
    nn_time = time.time() - start
    print(f"NeuralMultiTask: {nn_time:.4f}s ({len(preds_nn)/nn_time:.0f} preds/s)")

    # 4. Goals Regressor
    from stavki.models.neural.goals_regressor import GoalsRegressor
    gr = GoalsRegressor()
    gr.is_fitted = True
    # Setup dummy features corresponding to what's in data
    dummy_feats = [f"feature_{i}" for i in range(10)]
    gr.metadata["features"] = dummy_feats
    gr.features = dummy_feats
    
    gr.network = MagicMock()
    gr.network.return_value = (torch.tensor(np.random.exponential(1.5, n)), torch.tensor(np.random.exponential(1.2, n)))
    gr.feature_means = np.zeros(10)
    gr.feature_stds = np.ones(10)
    
    start = time.time()
    preds_gr = gr.predict(data)
    gr_time = time.time() - start
    print(f"GoalsRegressor: {gr_time:.4f}s ({len(preds_gr)/gr_time:.0f} preds/s)")

    return preds_dc, preds_lgb

if __name__ == "__main__":
    # Ensure torch is mocked if not present, but we are in environment with torch?
    # The script imports stavki.models... which imports torch.
    # We should handle the torch dependency in the script if needed, but assuming VM/Env has it.
    import torch
    
    # Run

    preds_dc, preds_lgb = benchmark_models()
    # Reuse data
    data = create_dummy_data(10000)
    benchmark_ensemble(data, preds_dc, preds_lgb)
    benchmark_backtest(data)
