# Prompt 2 — Test Rigor Report

**Gate**: `check_test_rigor.py --strict`  
**Status**: **PASS**  
**Date**: 2026-02-28T20:30

## Results

| Check | Result |
|-------|--------|
| Forbidden patterns (pytest.skip in except) | 0 found |
| Tests passed | 10 |
| Tests failed | 0 |
| Tests skipped | 0 |
| Tests xfailed | 0 |
| Tests xpassed | 0 |
| Tests errors | 0 |

## Changes Made

Removed all 4 `pytest.skip()` inside `except` blocks in `test_daily_prompt2_contract.py`:
- Line 77: `_map_features_to_model_inputs` isolation guard → removed try/except, kept try/finally
- Line 107: `_map_features_to_model_inputs` valid-case guard → removed entirely
- Line 162: `_find_value_bets` isolation guard → removed entirely
- Line 194: `_map_features_to_model_inputs` coverage-log guard → removed entirely

## Full Suite (Out-of-Scope Failures)

7 pre-existing failures unrelated to Prompt 2:
1. `tests/test_data/test_data_layer.py::TestTeamNormalization::test_basic_normalization`
2. `tests/test_data/test_data_layer.py::TestTeamNormalization::test_known_aliases`
3. `tests/test_data/test_data_layer.py::TestTeamNormalization::test_unicode_handling`
4. `tests/test_features/test_features.py::TestFeatureRegistry::test_fit_and_compute`
5. `tests/test_sportmonks_integration.py::test_unified_loader`
6. `tests/test_sportmonks_integration.py::test_live_predictions`
7. `tests/test_sportmonks_integration.py::test_backtesting_with_api_data`
