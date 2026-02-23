---
name: temporal-leakage-auditor
description: Use when constructing or modifying aggregate feature pipelines (Momentum, Form, Averages) to mathematically guarantee the future does not infect pre-match targets.
---

# Temporal Leakage Auditor

## Overview
"Target Leakage" or "Time-travel" is the single most catastrophic error a predictive AI can make. Feeding a model historical averages that accidentally include the score of the game you're trying to predict destroys betting backtests.

**Core principle:** ALWAYS rigorously sort and `.shift(1)` rolling arrays explicitly. Prove chronologically that `Time(Feature)` < `Time(Target)`.

## The Iron Law
```
NO PANDAS ARRAY MAY BE GROUPED OR SHIFTED WITHOUT BEING EXPLICITLY RE-SORTED BY `match_date` IMMEDIATELY PRIOR TO INVOCATION.
```

## When to Use This Skill
- When designing moving averages `.rolling().mean()` in `build_*.py` files.
- When creating momentum features or tracking streak logic.
- Use this ESPECIALLY if a new feature suddenly doubles model accuracy unexpectedly.

**Don't skip when:**
- "I already sorted the `match_date` yesterday when I built the parent table."
- You are joining supplementary databases using outer-merges.

## The Leakage Audit Phases

### Phase 1: Chronological Grounding
**BEFORE grouping or shifting temporal distributions:**
1. **Sort the Master Timeline**
   - Shifting an unsorted dataset randomizes the features mathematically, projecting arbitrary games against unrelated targets.
   ```python
   # ❌ Bad: Grouping directly
   df['avg_goals'] = df.groupby('team')['goals'].rolling(5).mean().values
   
   # ✅ Good: Defining chronological sequence boundaries strictly
   df = df.sort_values(['team_id', 'match_date'], ascending=True)
   df['avg_goals'] = df.groupby('team')['goals'].rolling(5).mean().shift(1).values
   ```

### Phase 2: Shift Diagnostics
1. **Enforce the `.shift(1)` Barrier**
   - An average encompassing the currently played target row creates implicit target leakage. Shifting by `1` isolates the previous `N` boundaries correctly.
2. **Proxy Variable Check**
   - Check the nature of the metric. Is "Total Corners Recorded at 90'" used to guess "Will Team A win"? These are essentially simultaneous outputs, resulting in deep target leakage.

### Phase 3: Visual Audit
1. **Target Row `__getitem__` Simulation**
   - To trust the logic, you must slice out an exact target and prove the `shift` aligns against previous dates.
   ```python
   # ✅ Good: Testing logic bounds individually
   print(df[df['team_id'] == 100][['match_date', 'goals', 'avg_goals']].tail(5))
   ```
2. **Cold-Start Fallback**
   - Confirm that Row `0` (the first game of the season for a team) returns a mathematically organic `NaN` or `0.0` due to the lack of history for the shift target to map backwards onto.

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "The `.shift()` function requires too many null drops, I'll just group without shifting to keep data clean."
- "I can merge this external referee rating Parquet onto the Gold target table without comparing timestamp drift."
- "The model reached `92%` validation accuracy on new inputs, this feature is incredible!" (It is almost certainly leaking target outcomes).

**ALL of these mean: STOP. You are hallucinating a time machine.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Ground** | `.sort_values('match_date')` | Historical continuum secured monotonically. |
| **2. Shift** | `.rolling(N).mean().shift(1)` logic | Pre-match aggregates completely isolate from target result row mathematically. |
| **3. Audit** | Visually inspecting a Pandas tail-slice `print` | Delta between Row `N` calculation and Row `N-1` scores align perfectly. |
