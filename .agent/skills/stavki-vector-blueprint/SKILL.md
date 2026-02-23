---
name: stavki-vector-blueprint
description: Use when appending new categorical features, historical floats, or contextual metrics to the STAVKI PyTorch or Baseline datasets.
---

# STAVKI Vector Blueprint

## Overview
Variables generated inside the Pandas logic must be surgically traced through the `StavkiDataset` layer, scaled contextually, and initialized mathematically in PyTorch. Dropping one sequential link creates batch shape mismatches.

**Core principle:** ALWAYS sync dimensions from DataFrames `datasets.py` empty tensors and PyTorch `context_dim` init blocks in parallel.

## The Iron Law
```
NO RAW CONTINUOUS FLOATS MAY BE FED TO DEEP NETWORKS WITHOUT EXPLICIT `MinMax` OR `StandardScaler` NORMALIZATION APPLIED FIRST.
```

## When to Use This Skill
- When designing `build_team_vectors.py` modifications.
- When generating newly engineered metrics like "Expected Goals Differential".
- Use this ESPECIALLY when switching the tensor length (e.g., from [1, 13] to [1, 17]).

**Don't skip when:**
- "I'm just adding one final momentum check."
- You believe "the fallback padding array in `datasets.py` will automatically expand."

## The Blueprint Phases

### Phase 1: Pandas Creation & Extraction
**DO NOT rely on inner joins if keys can drift.**
1. **Aggregating Logic**
   - Inject the localized metric into the `Silver` pipeline.
2. **Dimension Math Check**
   - Examine `.shape[1]` implicitly.
   ```python
   # ❌ Bad: Guessing vector shapes
   DeepInteractionNetwork(context_dim=15)
   
   # ✅ Good: Defining the explicit column math internally
   print(df_gold.shape[1])
   ```

### Phase 2: Padding & Scaling Synchronization
1. **Pytorch Preprocessing (`datasets.py`)**
   - You must intercept the target vector builder (`get_full_context()`, `get_mom_vec()`, etc.).
   ```python
   # ❌ Bad: Forgetting to expand the fallback vector for missing new teams
   if missing: return np.zeros(OLD_DIM, dtype=np.float32)
   
   # ✅ Good: Updating fallback arrays geometrically to the NEW_DIM length
   if missing: return np.zeros(NEW_DIM, dtype=np.float32)
   ```
2. **Standardization Protocol**
   - Verify that your new floating variable (e.g. `market_value=850000`) is tracked by the local `fit_transform` normalization matrices inside `prepare_matrices.py` or equivalent before entering PyTorch. Unscaled inputs completely destroy SGD gradient mapping.

### Phase 3: Network Topology Link
1. **Constructor Modification**
   - Update `stavki/models/deep_interaction.py`.
   - Update deployment calls inside `scripts/train_deep_interaction.py` to utilize `context_dim=NEW_DIM`.

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "The network knows how to reshape inputs automatically."
- "I'll just scale the target later if the training loss explodes."
- "I'll use an outer join and drop the `mid` primary key index to make the datasets merge cleanly."

**ALL of these mean: STOP. You are introducing silent shape-shifting bugs.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Create** | Dimension math, `.shape[1]` | New column builds fully without row drops. |
| **2. Pad/Scale** | Editing `datasets.py` `np.zeros()`, Scalers | `__getitem__` outputs identical lengths for cold-start cases. |
| **3. Link** | Modifying `context_dim` inside constructors | PyTorch Linear layer ingests the batch array successfully. |
