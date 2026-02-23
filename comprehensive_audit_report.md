# STAVKI Project Comprehensive Audit Report

This report consolidates the findings from our parallel Agent Team analysis of the `stavki_v2` project, explicitly targeting root flaws, API regressions, and surface-level implementation bugs across the entire stack.

---

## 🛑 Critical Issues Discovered

### 1. `SportMonksClient` Thread-Safety Violation (API Flaw)
**Domain:** External API Integration (`stavki/data/collectors/sportmonks.py`)
**Severity:** CRITICAL
- **The Bug:** The `SportMonksClient` uses a simple list `self._request_times = [t for t in self._request_times if t > minute_ago]` inside `_rate_limit_wait()` to manually track and throttle API queries to 50/minute.
- **The Crash:** In `DailyPipeline._enrich_matches()`, the client is deliberately passed into a `ThreadPoolExecutor` with `max_workers=10`. Because standard list mutations in Python are **not thread-safe**, the 10 concurrent threads simultaneously access and rewrite the `_request_times` list. This completely bypasses the rate limits, creating massive bursts that inevitably trigger catastrophic `429 Too Many Requests` API lockouts and pauses pipeline execution for 120 seconds.
- **Remedy:** Implement a `threading.Lock()` inside `_rate_limit_wait()` around the rate limit boundary checks, or utilize `threading.Semaphore`.

### 2. `EnsemblePredictor` Weight Ingestion Failure (Root Logic/Surface Formatter)
**Domain:** Prediction Architecture (`stavki/models/ensemble/predictor.py`)
**Severity:** CRITICAL
- **The Bug:** The system diligently optimizes per-league weights, storing them in `leagues.json` under legacy alias keys such as `"epl"`, `"laliga"`, and `"seriea"`. The `EnsemblePredictor.load_weights()` successfully parses these into `self.league_weights`. Let's assume `weights_for_epl` is mapped to key `"epl"`.
- **The Crash:** During live inference in `DailyPipeline.run()`, the matches stream in using the core `League` Enum values, e.g., `"soccer_epl"`. The loop passes `"soccer_epl"` to `EnsemblePredictor.get_weights("soccer_epl")`. Because `"soccer_epl"` != `"epl"`, the lookup fails silently without raising an exception.
- **The Impact:** The predictor catches the silent KeyError and falls back to `DEFAULT_WEIGHTS` (0 for most models). **The live system is operating completely un-optimized**, bypassing the customized per-league weights entirely.
- **Remedy:** Align the `leagues.json` saving/loading logic to map natively to `League.value` strings (`"soccer_epl"`), or inject a string normalization mapping inside `get_weights()`.

---

## ✅ Verified Systems (Working as Intended)

The following components were scrutinized for shortcuts or leakage but passed strict examination:

### 1. Temporal Splits & Data Leakage (Root ML Safety)
- **File:** `scripts/retrain_system.py`
- We verified that the 80/20 data splits strictly use time-series slicing (`train_df = df.iloc[:split_idx]`). No `test_train_split` randomizations are used, meaning no future games leak into the historical training loops.
- All odds columns representing external bookmaker probability metrics are accurately filtered out via RegEx masks before feeding CatBoost and Neural arrays, successfully forcing the models to extrapolate from intrinsic features only.
- Individual evaluators (`MultiTaskNetwork.fit` & `CatBoostModel.fit`) maintain the strict temporal ordering during early-stopping verification routines.

### 2. API Odds Extraction
- **File:** `stavki/data/collectors/odds_api.py`
- Safe fallbacks and exception monitoring are correctly implemented. `timeout=30` and `api_key` extractions are robust. Rate limiting here uses a simple `time.sleep` mechanism which operates sequentially so it is *not* vulnerable to the threading bugs seen in the SportMonks client. 

---

## 🧹 Surface Anomalies & Cleanup Targets

- **Exception Swallowing:** `EnsemblePredictor.load_weights()` uses an `except Exception as e:` block that only logs to `logger.error` before continuing execution. If the weights file is corrupted, it silently degrades to zeroes without hard-aborting the pipeline. 
- **OpenWeather API Extraneousness:** A previous checklist hinted at auditing OpenWeather API, but weather data is actively fetched entirely via `get_fixture_weather()` from SportMonks, making standalone weather endpoints obsolete logic.

## Recommended Next Steps
1. Deploy a fast-fix for the thread lock within `SportMonksClient._rate_limit_wait()`.
2. Rectify the `EnsemblePredictor` dictionary key mapping for leagues so optimization applies live.
