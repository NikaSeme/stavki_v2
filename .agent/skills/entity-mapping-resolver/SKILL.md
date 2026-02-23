---
name: entity-mapping-resolver
description: Use when the Daily STAVKI scanner fails to lock a match due to cross-API string or club identifier mismatches (e.g. OddsAPI to SportMonks translations).
---

# Entity Mapping Resolver

## Overview
STAVKI depends on perfect correlation between Live Odds feeds (`json`) and Historical API structures (`parquet`). Spelling discrepancies ("Man Utd" vs. "Manchester United") completely sever the model's access to the team's historical ELO.

**Core principle:** ALWAYS map specific, geographic identities sequentially via `difflib` bounds. NEVER overwrite a canonical root name.

## The Iron Law
```
NO "FUZZY KEY MATCHING" MAY OCCUR ACROSS GEOGRAPHIC BOUNDARIES. TWO CLUBS NAMED "WANDERERS" OR "CITY" IN DIFFERENT NATIONS WILL HALLUCINATE MODELS.
```

## When to Use This Skill
- When investigating the `bot.log` or `manual_scan.py` for "missing key" or "Unmapped Entity" warnings.
- When `fix_odds_api_mappings.py` fails to discover live games.
- Use this ESPECIALLY when live odds arrays generate `$0.00` stakes on major matches due to context separation.

**Don't skip when:**
- "It is obviously `Spurs`, I'll just hardcode a mapping in `daily.py`."
- "There is only one missing game, I can ignore it today."

## The Resolution Phases

### Phase 1: Mismatch Identification
**BEFORE executing naive generic matching:**
1. **Pinpoint the Geographic Boundary**
   - Identify the `country_id` or `league_id` the mismatched string originates from.
   ```python
   # ❌ Bad: Matching strings indiscriminately across the globe
   match = difflib.get_close_matches("Nacional") 
   # Returns mapping to Portugal instead of Uruguay...
   
   # ✅ Good: Slicing the candidate array strictly by country prior to string algorithms
   uk_teams = canonical_db[canonical_db.country == "England"]
   match = difflib.get_close_matches("Nottm Forest", uk_teams)
   ```

### Phase 2: Central Storage Injection
1. **Edit the `team_synonyms.json` Config**
   - Route the discovered stray string implicitly to its Master canonical name mapped by SportMonks.
   ```json
   // ✅ Good: Modifying central registries
   "Nott'm Forest": "Nottingham Forest",
   "Spurs": "Tottenham Hotspur"
   ```

### Phase 3: Auditing Pipeline
**You MUST prove the central fix restored the missing payload.**
1. **Run Validation Checks**
   - Execute the native bash verification module aggressively.
   ```bash
   # ✅ Good: Enforcing completeness
   python scripts/test_mappings.py
   ```
2. **Re-Test Live Feed**
   - Loop `manual_scan.py`. The "Unmapped Entity" warnings must disappear dynamically. Let the loop fail and retry mapping if unmapped lines persist.

## Red Flags - STOP and Follow Process
If you catch yourself thinking:
- "I'll just change the underlying SportMonks target string in the `Gold` database so it matches OddsAPI."
- "The match is 98% confident according to Levenshtein, but I don't need to check if the countries line up."
- "I can skip running `test_mappings.py` locally because it was just a comma fix."

**ALL of these mean: STOP. You are corrupting the alias graph permanently.**

## Quick Reference
| Phase | Key Activities | Success Criteria |
|-------|---------------|------------------|
| **1. Target** | `difflib` bound to `competition_id` / `country` | String is mathematically mapped without international overlap. |
| **2. Map** | Adding `{stray : root}` dict | `synonyms.json` is synced identically. |
| **3. Audit** | `test_mappings.py` execution | 100% fixture coverage restored inside `bot.log`. |
