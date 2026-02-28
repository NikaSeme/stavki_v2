# Prompt 1 — Truth Report Summary

**Scope**: Hotfix verification for `get_fixture_full` + `_fetch_recent_from_api`  
**Decision**: **GO**  
**Date**: 2026-02-28T18:32

## Claims

| # | Claim | Status | Critical | Evidence |
|---|-------|--------|----------|----------|
| 1 | `get_fixture_full` returns dict, never None | TRUE | ✅ | 9 callee tests pass, 22/22 focused tests pass |
| 2 | `get_fixture_full` callee test coverage | TRUE | ✅ | 12 test refs (contract gate) |
| 3 | `get_fixture_full` caller-path coverage | TRUE | ✅ | 2 caller refs + non-dict guard test |
| 4 | `_fetch_recent_from_api` callee test coverage | TRUE | ✅ | 9 test refs (contract gate), was 0 before fix |
| 5 | `_fetch_recent_from_api` caller-path test | TRUE | ✅ | 2 caller-path tests pass |
| 6 | Contract gate `--strict` passes | TRUE | ✅ | `"status": "pass"` |
| 7 | No new test failures | TRUE | ✅ | 133 pass, 7 fail (all match known pre-existing list) |
| 8 | No out-of-scope files modified | TRUE | — | scope report: `out_of_scope_new_files: []` |

## Contract Gate Warning (Transparent)

> **Warning**: `_fetch_recent_from_api: no caller refs found outside callee files`

This is a **known limitation of the static scan**, not a coverage gap. The only caller (`get_historical_data`) is inside `loader.py`, which is listed as a callee file. The gate's static scanner cannot detect intra-file callers. This is compensated by explicit caller-path tests:
- `test_get_historical_data_calls_fetch_recent` — verifies the call happens
- `test_get_historical_data_no_api_when_flag_false` — verifies it doesn't happen when disabled

## Gate Validation

```
check_truth_report.py --strict => PASS
  claims_total: 8
  critical_false: 0
  critical_unknown: 0
```

## Pre-Existing Failures (7, outside Prompt 1 scope)

| Test | Root Cause |
|------|-----------|
| `TestTeamNormalization::test_basic_normalization` | Normalization table incomplete |
| `TestTeamNormalization::test_known_aliases` | Same |
| `TestTeamNormalization::test_unicode_handling` | Same |
| `TestFeatureRegistry::test_fit_and_compute` | Feature registry issue |
| `test_unified_loader` | Stale keyword `start=` vs `start_date=` |
| `test_live_predictions` | Redis not running locally |
| `test_backtesting_with_api_data` | Same stale keyword |
