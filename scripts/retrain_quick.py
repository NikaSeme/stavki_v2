"""Quick retrain script for core models on enriched CSV."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    df = pd.read_csv("data/features_full.csv", low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    train_df = df.iloc[:int(len(df) * 0.7)].copy().reset_index(drop=True)
    logger.info(f"Using {len(train_df)} matches for training, {len(df.columns)} columns")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # 1. LightGBM_BTTS
    logger.info("Training LightGBM_BTTS...")
    from stavki.models.gradient_boost.btts_model import BTTSModel
    btts = BTTSModel(n_estimators=300, learning_rate=0.05)
    metrics = btts.fit(train_df, eval_ratio=0.2, early_stopping_rounds=30)
    btts.save(models_dir / "LightGBM_BTTS.pkl")
    logger.info(f"  BTTS metrics: {metrics}")

    # 2. LightGBM_1X2
    logger.info("Training LightGBM_1X2...")
    from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
    lgb = LightGBMModel(n_estimators=500, learning_rate=0.05)
    metrics = lgb.fit(train_df, eval_ratio=0.2, early_stopping_rounds=50)
    lgb.save(models_dir / "LightGBM_1X2.pkl")
    logger.info(f"  LightGBM 1X2 metrics: {metrics}")

    # 3. DixonColes
    logger.info("Training DixonColes...")
    from stavki.models.poisson.dixon_coles import DixonColesModel
    dc = DixonColesModel()
    dc.fit(train_df)
    dc.save(models_dir / "DixonColes.pkl")
    dc.save(models_dir / "dixon_coles.pkl")
    dc.save(models_dir / "poisson.pkl")
    logger.info("  DixonColes trained")

    # 4. CatBoost
    logger.info("Training CatBoost_1X2...")
    from stavki.models.catboost.catboost_model import CatBoostModel
    cb = CatBoostModel()
    metrics = cb.fit(train_df, eval_ratio=0.2)
    cb.save(models_dir / "catboost.pkl")
    logger.info(f"  CatBoost metrics: {metrics}")

    logger.info("All core models retrained on enriched data!")
    logger.info("Skipping NeuralMultiTask (very slow on CPU, optional)")

if __name__ == "__main__":
    main()
