---
name: catboost-hyper-tuner
description: Use when requested to tune STAVKI tree-based models (LightGBM, CatBoost) to prevent chronic predictive overfitting on noisy sports data payloads.
---

# CatBoost Hyper Tuner

## Overview
Machine learning on sports statistics is inherently noisy. Using standard deep Tree-searches (like grid-searching `depth=12`) always results in 99% training accuracy and catastrophic backtesting failure. 

**Core principle:** ALWAYS constrain tuning grids mathematically and measure performance purely against the Early Stopping validation slice.

## The Iron Law
```
NO TUNED CATBOOST OR LIGHTGBM MODEL MAY BE DEPLOYED IF A SINGLE FEATURE CLAIMS >80% OF THE SPLIT IMPORTANCE (DATA LEAKAGE).
```

## When to Use This Skill
- When tasked with "optimizing", "grid searching", or "tuning" the regressor/classifier layers in `scripts/train_tree_models.py`.
- When performance degrades across a historical block and the baseline requires adjustment.
- Use this ESPECIALLY when attempting hyperparameter random walks using Optuna.

**Don't skip when:**
- "The model is underfitting, let's just make the trees deeper."
- You are optimizing `learning_rate` locally on your desktop.

## The Tuning Phases

### Phase 1: Baseline Acquisition & Bounds
**BEFORE creating the grid iterator:**
1. **Log Existing Precision**
   - You must have a metric (`RMSE` or `Logloss`) to beat.
2. **Defensive Parameter Boundaries**
   - Noiseless data handles extreme shapes. Sports data does not.
   ```python
   # ❌ Bad: Allowing infinite depth searches
   param_grid = {'depth': [6, 10, 15]}
   
   # ✅ Good: Defining strict bounding boxes
   param_grid = {
       'learning_rate': [0.01, 0.05, 0.1],
       'depth': [4, 6, 8],
       'l2_leaf_reg': [1, 5, 10]
   }
   ```

### Phase 2: Evaluation Constraints
1. **Early Stopping Necessity**
   - Tuning without validation epochs guarantees hallucination. 
   ```python
   # ✅ Good: Enforcing stopping rounds
   model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)
   ```

### Phase 3: Deployment Auditing
1. **Importance Tracking**
   - Immediately output the `.get_feature_importance()` payload against the best model.
   - If a feature like `home_goals` receives a `.99` relative importance ratio predicting the Match Winner, you have leaked target data.
   - Print the delta metric explicitly comparing the New Tuned model against Old Baseline model to the user visually.

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "The model hits `0.00` MAE training loss, it's perfect!" (Catastrophic Overfitting).
- "I'll just try `depth=12` for a quick benchmark." (It will memorize the dataset completely).
- "I found the optimal params natively, no need to check how the exact features split computationally."

**ALL of these mean: STOP. You are creating a statistically invalid model.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Bounds** | Defining restricted param grids `depth<=8` | `Optuna`/`GridSearch` initializes securely within noise bounds. |
| **2. Fit** | Evaluation mapping, `early_stopping=50` | Out of fold generalization converges organically. |
| **3. Audit** | `.get_feature_importance()` extraction | Top features distribute evenly; target tracking < `80%`. |
