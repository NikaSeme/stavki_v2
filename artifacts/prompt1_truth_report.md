# Prompt 1 — Truth Report Summary

**Scope**: Hotfix verification for `get_fixture_full` + `_fetch_recent_from_api`  
**Decision**: **GO**  
**Date**: 2026-02-28T18:06

## Claims

| # | Claim | Status | Critical | Evidence |
|---|-------|--------|----------|----------|
| 1 | `get_fixture_full` returns dict, never None | TRUE | ✅ | 9 callee tests pass |
| 2 | `get_fixture_full` callee test coverage | TRUE | ✅ | 12 test refs (contract gate) |
| 3 | `get_fixture_full` caller-path coverage | TRUE | ✅ | 2 caller refs + guard test |
| 4 | `_fetch_recent_from_api` callee test coverage | TRUE | ✅ | 9 test refs (was 0) |
| 5 | `_fetch_recent_from_api` caller-path test | TRUE | ✅ | 2 caller-path tests pass |
| 6 | Contract gate `--strict` passes | TRUE | ✅ | `status: pass` |
| 7 | No new test failures | TRUE | ✅ | 133 pass, 7 known pre-existing |
| 8 | No out-of-scope files modified | TRUE | — | scope report clean |

## Gate Validation

```
check_truth_report.py --strict => PASS
  claims_total: 8
  critical_false: 0
  critical_unknown: 0
```

## Pre-Existing Failures (NOT Prompt 1 scope)

1. `test_basic_normalization` — normalization table incomplete
2. `test_known_aliases` — same
3. `test_unicode_handling` — same
4. `test_fit_and_compute` — feature registry issue
5. `test_unified_loader` — stale keyword `start=` vs `start_date=`
6. `test_live_predictions` — Redis not running locally
7. `test_backtesting_with_api_data` — same stale keyword
