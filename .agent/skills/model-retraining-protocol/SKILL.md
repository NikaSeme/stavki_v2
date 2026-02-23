---
name: model-retraining-protocol
description: Use when orchestrating the retraining of the STAVKI Deep Interaction Network, preventing fatal PyTorch dimension mismatches.
---

# Model Retraining Protocol

## Overview
Retraining the `DeepInteractionNetwork` requires perfectly synchronized steps across data generation, tensor dimension mapping, and PyTorch ingestion. Slight mismatches implicitly crash the network or ruin validation baseline tracking.

**Core principle:** ALWAYS ensure PyTorch embedding constraints scale relative to the dataset uniqueness values (e.g. `num_teams`) before starting the training runner.

## The Iron Law
```
NO PRODUCTION PYTORCH WEIGHTS MAY BE OVERWRITTEN UNLESS THE NEW VALIDATION LOSS DEMONSTRABLY BEATS THE PREVIOUS BASELINE, AND DIMENSIONALITY HAS BEEN PROVEN SYMMETRICAL.
```

## When to Use This Skill
- When a new column has been added to the Gold `.parquet` file.
- When expanding the number of teams, players, or leagues the system digests.
- Use this ESPECIALLY when altering `build_gold_pipeline.py`.

**Don't skip when:**
- "The new column is just a float, it probably won't break the embedding space."
- "I'm just quickly testing if the network learns."

## The Retraining Phases

### Phase 1: Feature Synchronization
**BEFORE touching `train_deep_interaction.py`:**

1. **Rebuild Gold Pipeline**
   - Ensure the new features properly integrate without crashing Pandas memory limits.
   ```bash
   # ✅ Good: Refreshing the parquet cache
   python scripts/build_gold_pipeline.py
   ```

2. **Calculate PyTorch Embeddings**
   - If the pipeline generated new string identifiers (like adding the German Bundesliga), the categorical space expanded.
   ```python
   # ❌ Bad: Guessing the dimension limits
   DeepInteractionNetwork(context_dim=15, num_teams=640)
   
   # ✅ Good: Tracing identical dimensional counts via Python directly
   print(len(full_ds.team_map)) # Outputs 643 teams
   ```

### Phase 2: Dimensional Synchronization
1. **Update `datasets.py` Allocations**
   - Locate fallback padding parameters inside `datasets.py` (`np.zeros(X)`).
   - If `context_dim` went from 13 to 17 in Pandas, update `X` to 17 here so missing data pads properly instead of crashing `StavkiDataset` slicing.
2. **Update Constructor**
   - Reflect dataset dimensional expansion into the actual PyTorch `DeepInteractionNetwork(context_dim=X)` layer.

### Phase 3: Training & Auditing
1. **Launch the Trainer**
   - Monitor the command line. PyTorch Early Stopping must halt the sequence organically.
2. **Benchmark Against Baseline**
   - Intercept the final validation loss. Compare it to previous baseline losses recorded locally.

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "The validation loss is higher, but the training loss feels really good."
- "I don't need to check `datasets.py`, I'm sure the fallback zero-arrays align."
- "The embedding ID sizes didn't change just because I added a new league."

**ALL of these mean: STOP. You are introducing silent tensor mismatch crashes.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Data Gen** | Gold pipeline build, Count embed sizes | Pandas merges clean without duplicate KeyErrors. |
| **2. Sync** | Edit `datasets.py` padding & PyTorch init shape | Linear intake batch vectors map cleanly `[B, NEW_DIM]` |
| **3. Training** | Val loss tracking | Validation loss < Previous Baseline |
