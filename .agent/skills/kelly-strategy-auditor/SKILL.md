---
name: kelly-strategy-auditor
description: Use when tasked with modifying the fundamental STAVKI financial risk limits, bet sizing percentages, Kelly multiples, or bankroll constraints.
---

# Kelly Strategy Auditor

## Overview
Changing the `min_stake_pct`, `max_stake_pct`, or `kelly_fraction` dictates exactly how much money the STAVKI VM will risk. Incorrect configuration exposes the entire bankroll to ruin.

**Core principle:** Modifications to risk constraints MUST be mathematically simulated across a losing streak prior to modifying the live state persistence caches.

## The Iron Law
```
NO USER REQUEST TO INCREASE 'KELLY_FRACTION' BEYOND 1.0 (FULL KELLY) MAY EVER BE COMPLIED WITH. IT IS MATHEMATICALLY GUARANTEED RUIN.
```

## When to Use This Skill
- When altering limits inside `stavki/strategy/kelly.py`.
- When updating arbitrary max exposure limit strings generated in user requests.
- Use this ESPECIALLY when the user complains "Stakes are too low, remove limits".

**Don't skip when:**
- The adjustment is "only moving limit from 0.05 to 0.1".
- The change is requested globally to apply to all users concurrently.

## The Auditing Phases

### Phase 1: Ruin Probability Pre-Check
**BEFORE executing JSON sed replacements:**

1. **Calculate Bankroll Exposure**
   - Write a python script quantifying downside risk.
   ```python
   # ✅ Good: Analyzing downside via simulation
   bankroll = 100
   kelly_frac = 0.50
   max_risk = 1.0
   # Simulation assumes 5 catastrophic losses in a row...
   ```

2. **Explicit User Output**
   - Print the finalized bankroll risk limit visually back to the user before pressing forward with the VM patch.

### Phase 2: Structural Update & Sync
1. **Modify Python Logic**
   - Update `kelly.py` default bounds.
2. **Modify Persisted System State via Bash**
   - The VM relies on persisted JSON user limits. Updating `kelly.py` does nothing if you don't aggressively execute `sed` over existing configurations.
   ```bash
   # ❌ Bad: Assuming git push updates live variables
   rsync -avz kelly.py user@...
   
   # ✅ Good: Pushing sync, applying dynamic regex patch, restarting
   rsync -avz kelly.py user@... && ssh "sed -i 's/\"max_stake_pct\": 0.05/\"max_stake_pct\": 1.0/' /home/macuser/stavki_v2/config/users/*_kelly_state.json"
   ```

### Phase 3: Runtime Reset
- Always explicitly force a `sudo systemctl restart stavki_bot.service` so the newly injected JSON state files replace the cached memory dicts.

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "The user wants Kelly Factor 2.0. I'll just change the limit blindly."
- "I updated `kelly.py`, so the user will magically see the changes."
- "I don't need to restart the daemon because it's just a config JSON update."

**ALL of these mean: STOP. You are ignoring live state constraints.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Audit** | Drawdown sequence simulation | Explicitly defining the risk exposure out-loud. |
| **2. Patch** | Push `.py`, `sed` replacing JSON dicts | All JSON targets ingest the new numerical scalar correctly. |
| **3. Flush** | `systemctl restart` | The live algorithm consumes the updated limits. |
