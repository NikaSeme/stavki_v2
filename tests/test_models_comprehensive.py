"""
Comprehensive Model Test Suite
==============================
Tests all models for correctness using proper pytest assertions.
Tests will FAIL in CI when models break, not silently pass.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime, timedelta
import sys
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.models.base import BaseModel, Market, Prediction
from stavki.models.poisson import DixonColesModel, GoalsMatrix
from stavki.models.ensemble import EnsemblePredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Generate realistic synthetic match data for testing."""
    np.random.seed(42)

    n_matches = 200
    n_teams = 20

    teams = [f"Team_{i}" for i in range(n_teams)]
    leagues = ["EPL", "LaLiga", "SerieA", "Bundesliga"]

    matches = []
    base_date = datetime(2024, 1, 1)

    # Team strength factors
    team_strength = {t: np.random.uniform(0.5, 1.5) for t in teams}

    for i in range(n_matches):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        league = np.random.choice(leagues)

        home_strength = team_strength[home]
        away_strength = team_strength[away]
        home_advantage = 1.2

        lambda_home = 1.3 * home_strength / away_strength * home_advantage
        lambda_away = 1.1 * away_strength / home_strength

        home_goals = np.random.poisson(lambda_home)
        away_goals = np.random.poisson(lambda_away)

        matches.append({
            "Date": base_date + timedelta(days=i // 5),
            "HomeTeam": home,
            "AwayTeam": away,
            "League": league,
            "FTHG": min(home_goals, 8),
            "FTAG": min(away_goals, 8),
            # Features
            "elo_home": 1500 + team_strength[home] * 200,
            "elo_away": 1500 + team_strength[away] * 200,
            "elo_diff": (team_strength[home] - team_strength[away]) * 200,
            "form_diff": np.random.normal(0, 2),
            "gf_diff": np.random.normal(0, 1),
            "ga_diff": np.random.normal(0, 1),
            # Odds
            "B365H": max(1.1, 3.0 / max(team_strength[home], 0.1)),
            "B365D": np.random.uniform(2.8, 4.5),
            "B365A": max(1.1, 3.0 / max(team_strength[away], 0.1)),
            "imp_home_norm": 0.0,
            "imp_draw_norm": 0.0,
            "imp_away_norm": 0.0,
        })

    df = pd.DataFrame(matches)

    # Compute implied probabilities
    total = 1 / df["B365H"] + 1 / df["B365D"] + 1 / df["B365A"]
    df["imp_home_norm"] = (1 / df["B365H"]) / total
    df["imp_draw_norm"] = (1 / df["B365D"]) / total
    df["imp_away_norm"] = (1 / df["B365A"]) / total

    # Full-time result
    df["FTR"] = np.where(
        df["FTHG"] > df["FTAG"], "H",
        np.where(df["FTHG"] < df["FTAG"], "A", "D")
    )

    return df


# ---------------------------------------------------------------------------
# Poisson model tests
# ---------------------------------------------------------------------------

class TestPoissonModel:
    """Test DixonColesModel (Poisson)."""

    def test_training(self, data):
        model = DixonColesModel()
        metrics = model.fit(data)
        assert "n_matches" in metrics
        assert "n_teams" in metrics
        assert metrics["n_matches"] > 0
        assert metrics["n_teams"] > 0

    def test_predictions(self, data):
        model = DixonColesModel()
        model.fit(data)
        preds = model.predict(data.iloc[:10])
        assert len(preds) > 0

        # Check multi-market output
        markets = set(p.market for p in preds)
        expected = {Market.MATCH_WINNER, Market.OVER_UNDER, Market.BTTS}
        assert markets >= expected, f"Missing markets: {expected - markets}"

    def test_probability_validity(self, data):
        model = DixonColesModel()
        model.fit(data)
        preds = model.predict(data.iloc[:10])

        for pred in preds[:5]:
            prob_sum = sum(pred.probabilities.values())
            assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum:.4f}, expected ~1.0"

    def test_goals_matrix(self):
        matrix = GoalsMatrix.from_lambdas(1.5, 1.2)
        h, d, a = GoalsMatrix.to_1x2(matrix)
        assert abs(h + d + a - 1.0) < 0.001, f"GoalsMatrix probs sum to {h+d+a:.4f}"

    def test_incremental_update(self, data):
        model = DixonColesModel()
        model.fit(data)
        # Should not raise
        model.update_match("Team_0", "Team_1", 2, 1)


# ---------------------------------------------------------------------------
# LightGBM model tests
# ---------------------------------------------------------------------------

class TestLightGBMModel:
    """Test LightGBM model."""

    # Features that exist in our synthetic test data
    TEST_FEATURES = [
        "elo_home", "elo_away", "elo_diff", "form_diff",
        "gf_diff", "ga_diff", "B365H", "B365D", "B365A",
        "imp_home_norm", "imp_draw_norm", "imp_away_norm",
    ]

    @pytest.fixture(autouse=True)
    def _check_lightgbm(self):
        pytest.importorskip("lightgbm", reason="LightGBM not installed")

    def test_training(self, data):
        from stavki.models.gradient_boost import LightGBMModel
        model = LightGBMModel(features=self.TEST_FEATURES)
        metrics = model.fit(data, eval_ratio=0.2)
        assert "eval_accuracy" in metrics
        assert metrics["eval_accuracy"] > 0

    def test_predictions(self, data):
        from stavki.models.gradient_boost import LightGBMModel
        model = LightGBMModel(features=self.TEST_FEATURES)
        model.fit(data, eval_ratio=0.2)
        preds = model.predict(data.iloc[:10])
        assert len(preds) > 0

    def test_feature_importance(self, data):
        from stavki.models.gradient_boost import LightGBMModel
        model = LightGBMModel(features=self.TEST_FEATURES)
        model.fit(data, eval_ratio=0.2)
        fi = model.get_feature_importance(top_n=5)
        assert len(fi) > 0


# ---------------------------------------------------------------------------
# CatBoost model tests
# ---------------------------------------------------------------------------

class TestCatBoostModel:
    """Test CatBoost model."""

    @pytest.fixture(autouse=True)
    def _check_catboost(self):
        pytest.importorskip("catboost", reason="CatBoost not installed")

    def test_training(self, data):
        from stavki.models.catboost import CatBoostModel
        model = CatBoostModel()
        metrics = model.fit(data, eval_ratio=0.2, verbose=0)
        assert "eval_accuracy" in metrics
        assert metrics["eval_accuracy"] > 0

    def test_predictions_calibration(self, data):
        from stavki.models.catboost import CatBoostModel
        model = CatBoostModel()
        model.fit(data, eval_ratio=0.2, verbose=0)
        preds = model.predict(data.iloc[:10])
        assert len(preds) > 0

        # Check calibration — max prob should be > random (0.33)
        avg_max_prob = np.mean([max(p.probabilities.values()) for p in preds])
        assert avg_max_prob > 0.33

    def test_feature_importance(self, data):
        from stavki.models.catboost import CatBoostModel
        model = CatBoostModel()
        model.fit(data, eval_ratio=0.2, verbose=0)
        fi = model.get_feature_importance(top_n=5)
        assert len(fi) > 0


# ---------------------------------------------------------------------------
# Neural model tests
# ---------------------------------------------------------------------------

class TestNeuralModels:
    """Test Neural network models."""

    @pytest.fixture(autouse=True)
    def _check_torch(self):
        pytest.importorskip("torch", reason="PyTorch not installed")

    def test_multitask_training(self, data):
        from stavki.models.neural import MultiTaskModel
        model = MultiTaskModel(n_epochs=10, hidden_dim=64)
        metrics = model.fit(data)
        assert "accuracy_1x2" in metrics
        assert metrics["accuracy_1x2"] > 0

    def test_multitask_multimarket(self, data):
        from stavki.models.neural import MultiTaskModel
        model = MultiTaskModel(n_epochs=10, hidden_dim=64)
        model.fit(data)
        preds = model.predict(data.iloc[:5])
        markets = set(p.market.value for p in preds)
        assert len(markets) >= 2, f"Expected multiple markets, got {markets}"

    def test_goals_regressor(self, data):
        from stavki.models.neural import GoalsRegressor
        model = GoalsRegressor(n_epochs=10)
        metrics = model.fit(data)
        assert "mae_home" in metrics
        assert metrics["mae_home"] > 0


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------

class TestEnsemble:
    """Test ensemble predictor."""

    def test_predictions(self, data):
        poisson = DixonColesModel()
        poisson.fit(data)

        ensemble = EnsemblePredictor(models={"DixonColes": poisson})
        preds = ensemble.predict(data.iloc[:10])
        assert len(preds) > 0

    def test_weight_optimization(self, data):
        poisson = DixonColesModel()
        poisson.fit(data)

        ensemble = EnsemblePredictor(models={"DixonColes": poisson})
        weights = ensemble.optimize_weights(data.iloc[:50], Market.MATCH_WINNER)
        assert isinstance(weights, dict)


# ---------------------------------------------------------------------------
# Standalone runner (for manual execution)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

