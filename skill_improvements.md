# Deep Analysis & Improvement Plan for STAVKI Skills

This document details the rigorous secondary analysis performed on the 8 custom skills to ensure perfect compliance with the 5-rule framework, addressing logic gaps, and improving verifiable steps.

## 1. `cloud-vm-sync`
**Deficiencies:**
- **Goal:** Clear, but lacks a rollback objective if the push crashes the live environment.
- **Prohibitions:** Misses a prohibition against ignoring environment variables or `.env` files.
- **Steps:** `python -m py_compile` assumes all synced files are Python. What if it's a `.json` or `.sh` file? The steps are not robust enough to context. It lacks a rollback step.
**Fixes:**
- Add JSON validation (`python -m json.tool`) and Bash syntax checks (`bash -n`).
- Add a mandatory rollback command step in case `systemctl status` fails.

## 2. `model-retraining-protocol`
**Deficiencies:**
- **Steps:** Misses a massive PyTorch specific constraint—categorical embedding dimensions. If `build_gold_pipeline.py` introduces new entities (like a promoted team), `num_teams` in the PyTorch dataset increases, requiring an embedding layer resize.
- **Verification:** Doesn't explicitly state how to get the previous validation loss.
**Fixes:**
- Add a `DO NOT` rule about training with mismatched categorical embedding sizes.
- Insert a verification step to print and compare `num_players`, `num_teams`, and `num_leagues` against the model's `Embedding` initialization.

## 3. `kelly-strategy-auditor`
**Deficiencies:**
- **Prohibitions:** Lacks a strict mathematical ceiling prohibition (e.g., Kelly > 1.0 is mathematically ruinous).
- **Steps:** "calculate what happens" is vague. The auditor needs a verifiable formula or a Python script reference to print the exact drawdown risk.
**Fixes:**
- Enforce that the Kelly Fraction input MUST NEVER exceed 1.0. 
- Require the creation of a temporary Python script to simulate 10 losses in a row at the new stake limit.

## 4. `stavki-data-sleuth`
**Deficiencies:**
- **Prohibitions:** Focuses on numerical imputation (median/zero) but ignores Categorical NaNs.
- **Steps:** Fails to address categorical variables (e.g., missing Referee IDs) which will break LightGBM natively if unhandled.
**Fixes:**
- Add specific steps for Categorical tracking: replace `NaN` with `"Unknown"` or `-1` explicitly so Tree models don't crash.

## 5. `catboost-hyper-tuner`
**Deficiencies:**
- **Prohibitions:** Ignores feature importance. Overfitting often manifests as one feature getting 99% importance.
- **Verification:** "Outperforms" is vague. Must specify metrics (`Logloss` or `RMSE`).
**Fixes:**
- Add a rule to reject models where a single feature dictates >80% of the model split decisions.
- Specify exact target metrics for the success criteria.

## 6. `stavki-vector-blueprint`
**Deficiencies:**
- **Prohibitions:** Missing normalization/scaling constraints. Neural networks mathematically require features to be scaled, usually via `StandardScaler`.
- **Steps:** Fails to instruct the builder to append the new feature to the scaling pipeline before feeding the Deep Network.
**Fixes:**
- Add a strict requirement to check `normalization` layers. Appending raw unscaled data like `market_value_euros=50000000` alongside `team_elo=1500` destroys gradients.

## 7. `entity-mapping-resolver`
**Deficiencies:**
- **Prohibitions:** Fuzzy matching purely on names is dangerous in global sports ("Nacional" in Uruguay vs Portugal).
- **Steps:** Lacks a verification check for the country/league boundary.
**Fixes:**
- Add a strict prohibition against mapping cross-league entities based solely on Levenshtein distance.
- Add a step to cross-reference the `competition_id` or `country` tag during mapping.

## 8. `temporal-leakage-auditor`
**Deficiencies:**
- **Prohibitions:** Missing the Golden Rule of time-series—the DataFrame MUST be strictly sorted by date before calling `.shift(1)`. If not sorted, shift mixes random games.
- **Steps:** Fails to warn against indirect target leakage (e.g., using post-match events like "Total Corners" to predict Match Winner).
**Fixes:**
- Command the auditor to explicitly execute `df = df.sort_values('match_date')` as the very first operation.
- Add a test case demonstrating target leakage rejection.
