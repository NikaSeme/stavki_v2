"""
CatBoost Hyperparameter Tuning Script
======================================
Follows catboost-hyper-tuner skill constraints:
  - depth ≤ 8 (no infinite depth searches on noisy sports data)
  - early_stopping_rounds = 50 (mandatory)
  - Feature importance audit: FAIL if any feature > 80% split importance

Usage:
  # Baseline only (no tuning):
  python scripts/train_tree_models.py --n-trials 0

  # Full Optuna tuning (30 trials):
  python scripts/train_tree_models.py --n-trials 30

  # Custom output:
  python scripts/train_tree_models.py --n-trials 30 --output models/catboost_tuned.pkl
"""

import argparse
import logging
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Skill-mandated parameter bounds (Iron Law)
# depth ≤ 8, no unrestricted grids
# ──────────────────────────────────────────────
PARAM_BOUNDS = {
    "learning_rate":       (0.01, 0.1),
    "depth":               (3, 8),         # max 8 per skill
    "l2_leaf_reg":         (1.0, 10.0),
    "random_strength":     (0.5, 3.0),
    "bagging_temperature": (0.5, 2.0),
    "iterations":          (300, 800),
}

# Leakage threshold (catboost-hyper-tuner Iron Law)
MAX_FEATURE_IMPORTANCE_PCT = 80.0


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare data with date sorting."""
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}...")
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, low_memory=False)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
        df = df.sort_values("Date").reset_index(drop=True)

    logger.info(f"Loaded {len(df)} matches, {len(df.columns)} columns")
    return df


def train_catboost(df: pd.DataFrame, **kwargs) -> tuple:
    """Train a CatBoostModel and return (model, metrics)."""
    from stavki.models.catboost.catboost_model import CatBoostModel

    model = CatBoostModel(**kwargs)
    metrics = model.fit(df, eval_ratio=0.20, early_stopping_rounds=50, verbose=0)
    return model, metrics


def audit_feature_importance(model, label: str = "Model") -> bool:
    """
    Check Iron Law: no single feature may claim > 80% importance.
    Returns True if audit PASSES (no leakage), False otherwise.
    """
    importances = model.get_feature_importance(top_n=50)
    if not importances:
        logger.warning(f"[{label}] No feature importance data available")
        return True

    total = sum(v for _, v in importances)
    if total == 0:
        return True

    logger.info(f"\n{'─'*60}")
    logger.info(f"Feature Importance Audit — {label}")
    logger.info(f"{'─'*60}")

    passed = True
    for rank, (feat, imp) in enumerate(importances[:15], 1):
        pct = (imp / total) * 100
        flag = " ⚠️  LEAKAGE" if pct > MAX_FEATURE_IMPORTANCE_PCT else ""
        logger.info(f"  {rank:2d}. {feat:<40s}  {pct:5.1f}%{flag}")
        if pct > MAX_FEATURE_IMPORTANCE_PCT:
            passed = False

    if passed:
        logger.info("  ✅ PASSED — No feature exceeds 80% importance")
    else:
        logger.error("  ❌ FAILED — Feature leakage detected (>80%)")

    return passed


def run_baseline(df: pd.DataFrame) -> dict:
    """Train with defaults and return baseline metrics."""
    logger.info("\n" + "═" * 60)
    logger.info("Phase 1: BASELINE — Training CatBoost with defaults")
    logger.info("═" * 60)
    logger.info("  depth=6, lr=0.05, l2_leaf_reg=3.0, iterations=500")

    start = datetime.now()
    model, metrics = train_catboost(df)
    elapsed = (datetime.now() - start).total_seconds()

    logger.info(f"\nBaseline Results ({elapsed:.1f}s):")
    logger.info(f"  Train accuracy:  {metrics['train_accuracy']:.4f}")
    logger.info(f"  Eval accuracy:   {metrics['eval_accuracy']:.4f}")
    logger.info(f"  Train log-loss:  {metrics['train_log_loss']:.4f}")
    logger.info(f"  Eval log-loss:   {metrics['eval_log_loss']:.4f}  ← metric to beat")
    logger.info(f"  Best iteration:  {metrics['best_iteration']}")

    audit_feature_importance(model, "Baseline")
    return {"model": model, "metrics": metrics}


def run_optuna(df: pd.DataFrame, n_trials: int, baseline_metrics: dict) -> dict:
    """Run Optuna search within skill-mandated bounds."""
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info("\n" + "═" * 60)
    logger.info(f"Phase 2: OPTUNA — {n_trials} trials within bounded grid")
    logger.info("═" * 60)
    logger.info(f"  depth:               [{PARAM_BOUNDS['depth'][0]}, {PARAM_BOUNDS['depth'][1]}]")
    logger.info(f"  learning_rate:       [{PARAM_BOUNDS['learning_rate'][0]}, {PARAM_BOUNDS['learning_rate'][1]}]")
    logger.info(f"  l2_leaf_reg:         [{PARAM_BOUNDS['l2_leaf_reg'][0]}, {PARAM_BOUNDS['l2_leaf_reg'][1]}]")
    logger.info(f"  random_strength:     [{PARAM_BOUNDS['random_strength'][0]}, {PARAM_BOUNDS['random_strength'][1]}]")
    logger.info(f"  bagging_temperature: [{PARAM_BOUNDS['bagging_temperature'][0]}, {PARAM_BOUNDS['bagging_temperature'][1]}]")
    logger.info(f"  iterations:          [{PARAM_BOUNDS['iterations'][0]}, {PARAM_BOUNDS['iterations'][1]}]")

    best_result = {"eval_log_loss": float("inf"), "params": {}, "metrics": {}, "model": None}

    def objective(trial):
        params = {
            "learning_rate":       trial.suggest_float("learning_rate", *PARAM_BOUNDS["learning_rate"], log=True),
            "depth":               trial.suggest_int("depth", *PARAM_BOUNDS["depth"]),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg", *PARAM_BOUNDS["l2_leaf_reg"]),
            "random_strength":     trial.suggest_float("random_strength", *PARAM_BOUNDS["random_strength"]),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *PARAM_BOUNDS["bagging_temperature"]),
            "iterations":          trial.suggest_int("iterations", *PARAM_BOUNDS["iterations"]),
        }

        model, metrics = train_catboost(df, **params)
        eval_ll = metrics["eval_log_loss"]

        logger.info(
            f"  Trial {trial.number:3d} | "
            f"depth={params['depth']} lr={params['learning_rate']:.4f} "
            f"l2={params['l2_leaf_reg']:.1f} | "
            f"eval_ll={eval_ll:.4f} eval_acc={metrics['eval_accuracy']:.4f}"
        )

        if eval_ll < best_result["eval_log_loss"]:
            best_result["eval_log_loss"] = eval_ll
            best_result["params"] = params
            best_result["metrics"] = metrics
            best_result["model"] = model

        return eval_ll

    study = optuna.create_study(direction="minimize", study_name="catboost_tuning")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info(f"\n  Best trial: #{study.best_trial.number}")
    logger.info(f"  Best eval log-loss: {study.best_value:.4f}")
    logger.info(f"  Best params: {json.dumps(study.best_params, indent=4)}")

    return best_result


def compare_and_save(
    baseline: dict,
    tuned: dict,
    output_path: Path,
) -> bool:
    """Phase 3: Compare, audit, and save the best model."""
    logger.info("\n" + "═" * 60)
    logger.info("Phase 3: DEPLOYMENT AUDIT")
    logger.info("═" * 60)

    bl_ll = baseline["metrics"]["eval_log_loss"]
    tu_ll = tuned["metrics"]["eval_log_loss"]
    delta = tu_ll - bl_ll

    bl_acc = baseline["metrics"]["eval_accuracy"]
    tu_acc = tuned["metrics"]["eval_accuracy"]

    logger.info(f"\n  {'Metric':<25s}  {'Baseline':>10s}  {'Tuned':>10s}  {'Delta':>10s}")
    logger.info(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*10}")
    logger.info(f"  {'Eval Log-Loss':<25s}  {bl_ll:10.4f}  {tu_ll:10.4f}  {delta:+10.4f}")
    logger.info(f"  {'Eval Accuracy':<25s}  {bl_acc:10.4f}  {tu_acc:10.4f}  {tu_acc - bl_acc:+10.4f}")

    if delta < 0:
        logger.info(f"\n  ✅ Tuned model IMPROVED log-loss by {abs(delta):.4f}")
    elif delta == 0:
        logger.info(f"\n  ➖ Tuned model matched baseline exactly")
    else:
        logger.warning(f"\n  ⚠️  Tuned model WORSE by {delta:.4f} — consider keeping baseline")

    # Feature importance audit on tuned model
    audit_passed = audit_feature_importance(tuned["model"], "Tuned (Best)")

    if not audit_passed:
        logger.error("\n❌ ABORTING SAVE — Feature leakage detected. Model is statistically invalid.")
        return False

    # Save
    tuned["model"].save(output_path)
    logger.info(f"\n  💾 Saved tuned model to {output_path}")

    # Also save params for reference
    params_path = output_path.with_suffix(".json")
    with open(params_path, "w") as f:
        json.dump(
            {
                "best_params": tuned["params"],
                "baseline_metrics": baseline["metrics"],
                "tuned_metrics": tuned["metrics"],
                "delta_log_loss": float(delta),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    logger.info(f"  📄 Saved params to {params_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="CatBoost Hyperparameter Tuning (Skill-Constrained)")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials (0 = baseline only)")
    parser.add_argument("--data-path", type=str, default="data/features_full.csv", help="Path to features CSV/parquet")
    parser.add_argument("--output", type=str, default="models/catboost_tuned.pkl", help="Output model path")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    logger.info("🐱 CatBoost Hyperparameter Tuning")
    logger.info(f"   Data:     {data_path}")
    logger.info(f"   Trials:   {args.n_trials}")
    logger.info(f"   Output:   {output_path}")

    # Load data
    df = load_data(data_path)

    # Phase 1: Baseline
    baseline = run_baseline(df)

    if args.n_trials == 0:
        logger.info("\n✅ Baseline captured. No tuning requested (--n-trials 0).")
        audit_passed = audit_feature_importance(baseline["model"], "Baseline")
        if not audit_passed:
            sys.exit(1)
        return

    # Phase 2: Optuna
    tuned = run_optuna(df, args.n_trials, baseline["metrics"])

    # Phase 3: Compare, audit, save
    success = compare_and_save(baseline, tuned, output_path)
    if not success:
        sys.exit(1)

    logger.info("\n✅ CatBoost tuning complete!")


if __name__ == "__main__":
    main()
