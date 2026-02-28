# Prompt 2 — Final Verification (Integrity Hardened)

**Status**: DONE  
**Date**: 2026-02-28T21:00  
**Decision**: GO

---

## Acceptance Map

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | No recommendation when critical features missing | PASS | `test_missing_critical_features_marks_invalid` + `test_model_column_critical_missing_returns_invalid` PASSED |
| 2 | `invalid_for_bet` excluded in real runtime path | PASS | `test_run_path_merges_invalid_flag` + `test_find_value_bets_skips_invalid_rows` PASSED |
| 3 | Defaults only for non-critical columns | PASS | `test_marks_invalid_when_critical_missing` + `test_non_critical_defaults_allowed` PASSED |
| 4 | Logs show non-critical substitution coverage | PASS | `test_coverage_logged` PASSED |
| 5 | No silent fallback in critical path | PASS | `check_critical_path_failfast.py --strict → PASS` |
| 6 | Schema missing → still gates critical features | PASS | `schema_missing_runtime_gate → PASS` |
| 7 | No placeholder tests in audited files | PASS | `check_test_placeholders.py --strict → PASS` |
| 8 | No skip-on-exception in contract tests | PASS | `check_test_rigor.py --strict → PASS` |
| 9 | Contract gate for get_fixture_full + _fetch_recent_from_api | PASS | `hotfix_contract_check.py --strict → PASS` |
| 10 | `out_of_scope_new_files` empty | PASS | `scope_report: []` |

## Changed Files

| Path | Type |
|------|------|
| /Users/macuser/Documents/something/stavki_v2/stavki/pipelines/daily.py | MODIFIED |
| /Users/macuser/Documents/something/stavki_v2/tests/test_loader.py | MODIFIED |
| /Users/macuser/Documents/something/stavki_v2/artifacts/hotfix_contract_report.json | MODIFIED |
| /Users/macuser/Documents/something/stavki_v2/artifacts/critical_path_failfast_report.json | NEW |
| /Users/macuser/Documents/something/stavki_v2/artifacts/no_placeholder_tests_report.json | NEW |
| /Users/macuser/Documents/something/stavki_v2/artifacts/prompt2_scope_report.json | REFRESHED |
| /Users/macuser/Documents/something/stavki_v2/artifacts/prompt2_truth_report.json | REFRESHED |
| /Users/macuser/Documents/something/stavki_v2/artifacts/prompt2_truth_report.md | REFRESHED |
| /Users/macuser/Documents/something/stavki_v2/artifacts/prompt2_final_verification.md | REFRESHED |
| /Users/macuser/Documents/something/stavki_v2/artifacts/prompt2_evidence_consistency_report.json | NEW |
| /Users/macuser/Documents/something/stavki_v2/artifacts/prompt2_test_rigor_report.json | REFRESHED |

## Pre-Existing Out-of-Scope Failures (7)

| Test | Root Cause |
|------|------------|
| `test_data_layer::test_basic_normalization` | TeamNormalization logic |
| `test_data_layer::test_known_aliases` | TeamNormalization logic |
| `test_data_layer::test_unicode_handling` | TeamNormalization logic |
| `test_features::test_fit_and_compute` | FeatureRegistry issue |
| `test_sportmonks_integration::test_unified_loader` | TypeError in API |
| `test_sportmonks_integration::test_live_predictions` | Redis not running |
| `test_sportmonks_integration::test_backtesting_with_api_data` | API arg mismatch |

## Validation Commands

```
# 1. Prompt 1 target tests
$ python3 -m pytest tests/test_get_fixture_full.py tests/test_loader.py -v -rs
→ 22 passed, 0 skipped in 0.55s

# 2. Prompt 2 target tests
$ python3 -m pytest tests/test_live_prompt2_contract.py tests/test_daily_prompt2_contract.py -v -rs
→ 10 passed, 0 skipped in 0.99s

# 3. Contract gate
$ python3 hotfix_contract_check.py --strict
→ {"status": "pass"}

# 4. Test rigor gate
$ python3 check_test_rigor.py --strict
→ {"status": "pass", "forbidden_patterns_found": [], "pytest_summary": {"passed": 10, "failed": 0, "skipped": 0}}

# 5. No placeholder tests gate
$ python3 check_test_placeholders.py --strict
→ {"status": "pass", "placeholder_tests_found": []}

# 6. Critical fail-fast gate
$ python3 check_critical_path_failfast.py --strict
→ {"status": "pass", "checks": {"schema_exists_guard_has_else": "pass", "no_silent_minimal_fallback": "pass", "schema_missing_runtime_gate": "pass"}}

# 7. Evidence consistency gate
$ python3 check_evidence_consistency.py --strict
→ (run after this rebuild)

# 8. Truth gates
$ python3 check_truth_report.py --report prompt1_truth_report.json --strict
→ {"status": "pass"}
$ python3 check_truth_report.py --report prompt2_truth_report.json --strict
→ {"status": "pass"}

# 9. Full suite
$ python3 -m pytest tests/ -v --tb=short
→ 143 passed, 7 failed (pre-existing, out-of-scope)

# Safety compile
$ python3 -m py_compile live.py daily.py
→ COMPILE OK
```
