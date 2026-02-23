---
name: stavki-data-sleuth
description: Use when debugging broken STAVKI Pandas DataFrames, handling NaNs in upstream pipelines, or repairing categorical data mismatches in Silver-to-Gold parquet builds.
---

# STAVKI Data Sleuth

## Overview
Broken features, missing inputs, and `KeyError` logs are symptoms of corrupted Pandas logic or scraped API failures. Using arbitrary `.fillna(0)` across diverse distributions guarantees downstream Gradient Boosting and PyTorch crashes. 

**Core principle:** ALWAYS identify the shape and temporal boundary of the missing data before imputing. 

## The Iron Law
```
NO DATAFRAME `NaN` MAY BE IMPUTED UNTIL ITS DISTRIBUTION (CATEGORICAL, CHRONOLOGICAL, OR STRUCTURAL) HAS EXPLICITLY BEEN IDENTIFIED.
```

## When to Use This Skill
- When investigating the STAVKI Daily Scanning log emitting "Missing `mid`/`tid`" warnings.
- When `build_team_vectors.py` throws alignment crashes.
- Use this ESPECIALLY when model accuracy arbitrarily drops due to silent NaN propagation inside LightGBM.

**Don't skip when:**
- "It's just one missing ELO value, I'll `fillna(0)` to pass the test."
- You see `KeyError` matching on SportMonks arrays.

## The Investigative Phases

### Phase 1: Diagnostics and Scope Tracing
**BEFORE modifying imputation pipelines:**
1. **Analyze Corruptions by Domain**
   - Write standard Pandas interrogation scripts dynamically.
   ```python
   # ❌ Bad: Guessing the missing data layer
   df = pd.read_parquet('team_vectors.parquet')
   df.fillna(0, inplace=True)
   
   # ✅ Good: Defining scope and temporal boundary natively
   print(df.isna().sum())
   print(df['match_date'].min(), df['match_date'].max())
   ```
2. **Examine Upstream Silver Dependencies**
   - If missing in `Gold`, check `Silver` explicitly. Do not mask scrape failures with imputation logic.

### Phase 2: Domain-Specific Imputation Rules
**DO NOT arbitrarily mix `zeros` vs `medians`.**
1. **Chronological / Form Logic**
   - Natural NaNs occur for newly promoted teams.
   ```python
   # ✅ Good: Padding form/momentum with 0.0 correctly
   df['trailing_goals'].fillna(0.0)
   ```
2. **Structural & Baseline Metrics**
   - NaNs in ELO ratings break scaling constraints.
   ```python
   # ✅ Good: Median imputation for ratings
   df['team_elo'].fillna(df['team_elo'].median())
   ```
3. **Categorical Targets**
   - Native categorical trees crash if identifiers are null.
   ```python
   # ❌ Bad: Ignoring strings
   
   # ✅ Good: Creating an explicit Unknown branch
   df['referee_name'].fillna('Unknown_Ref')
   ```

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "I'll just forward-fill (`ffill()`) the remaining ELO gaps." (This risks bleeding future outcomes into the past).
- "I'm going to impute the entire DataFrame with zeroes just to make the test pass."
- "The missing Referee ID doesn't matter, Tree models can handle missing values natively." (They do, but only logically).

**ALL of these mean: STOP. You are injecting statistical hallucinations.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Trace** | `isna().sum()`, upstream verification | The exact corrupted API boundary is identified. |
| **2. Fix** | Apply Medians, Zeroes, or Strings contextually | `build_gold_pipeline.py` executes cleanly. |
| **3. Audit** | Resample `match_id` directly in Pandas | Specific output row holds expected integer / string proxy. |
