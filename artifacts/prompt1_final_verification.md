# Prompt 1 — Final Verification Report

**Date**: 2026-02-28T18:06  
**Scope**: `get_fixture_full` fix + `_fetch_recent_from_api` contract coverage  
**Decision**: **GO** — all gates pass

---

## Acceptance Map

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | Focused tests pass | **PASS** | `python3 -m pytest tests/test_get_fixture_full.py tests/test_loader.py -v` → 22 passed in 0.52s |
| 2 | Contract gate strict | **PASS** | `hotfix_contract_check.py --strict` → `"status": "pass"` |
| 3 | Full suite — no new failures | **PASS** | `python3 -m pytest tests/ -v --tb=short` → 133 pass, 7 fail (all pre-existing) |
| 4 | Truth report valid | **PASS** | `check_truth_report.py --strict` → `"status": "pass"`, 8 claims, 0 critical false |
| 5 | Scope clean | **PASS** | `prompt1_scope_report.json` → `out_of_scope_new_files: []` |

## Exact Commands Executed

```bash
# Step 1 — Baseline
git status --porcelain > artifacts/pre_task_status.txt

# Step 2 — Focused tests
python3 -m pytest tests/test_get_fixture_full.py tests/test_loader.py -v
# Result: 22 passed, 0 failed, 1 warning in 0.52s

# Step 3 — Contract gate
python3 ~/.gemini/antigravity/skills/antigravity-python-contract-gate/scripts/hotfix_contract_check.py \
  --repo /Users/macuser/Documents/something/stavki_v2 \
  --functions get_fixture_full _fetch_recent_from_api \
  --tests-root tests \
  --callee-files stavki/data/collectors/sportmonks.py stavki/data/loader.py \
  --report artifacts/hotfix_contract_report.json --strict
# Result: {"status": "pass"}

# Step 4 — Full suite
python3 -m pytest tests/ -v --tb=short
# Result: 133 passed, 7 failed (all pre-existing), 2 warnings in 18.34s

# Step 5 — Post-task status
git status --porcelain > artifacts/post_task_status.txt

# Step 6 — Scope report
# Built artifacts/prompt1_scope_report.json => out_of_scope_new_files: []

# Step 7 — Truth report validation
python3 ~/.gemini/antigravity/skills/antigravity-truth-report-gate/scripts/check_truth_report.py \
  --report artifacts/prompt1_truth_report.json \
  --min-evidence 1 --require-two-evidence-for-critical --strict
# Result: {"status": "pass", "claims_total": 8, "critical_false": 0, "critical_unknown": 0}
```

## Mandatory Artifacts Produced

| File | Content |
|------|---------|
| `artifacts/pre_task_status.txt` | Git baseline before task |
| `artifacts/post_task_status.txt` | Git status after task |
| `artifacts/hotfix_contract_report.json` | Contract gate report |
| `artifacts/prompt1_scope_report.json` | Scope gate report |
| `artifacts/prompt1_truth_report.json` | Truth report (machine) |
| `artifacts/prompt1_truth_report.md` | Truth report (human) |
| `artifacts/prompt1_final_verification.md` | This file |

## Pre-Existing Failures (7, all outside Prompt 1 scope)

| Test | Root Cause |
|------|-----------|
| `test_basic_normalization` | Normalization table incomplete |
| `test_known_aliases` | Same |
| `test_unicode_handling` | Same |
| `test_fit_and_compute` | Feature registry issue |
| `test_unified_loader` | Stale API keyword (`start=` vs `start_date=`) |
| `test_live_predictions` | Redis not running locally |
| `test_backtesting_with_api_data` | Same stale keyword |
