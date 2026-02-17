"""
Enrich Features CSV — Generate missing BTTS-critical features.

Reads `data/features_full.csv`, computes 10 missing features from raw match
data using only backward-looking rolling windows (no data leakage), then
writes the enriched data back.

Features generated:
  1. form_home_ga          — rolling avg goals conceded (home team, last 5 matches)
  2. form_away_ga          — rolling avg goals conceded (away team, last 5 matches)
  3. attack_strength_home  — home team scoring rate vs league average
  4. attack_strength_away  — away team scoring rate vs league average
  5. defense_strength_home — home team conceding rate vs league average
  6. defense_strength_away — away team conceding rate vs league average
  7. advanced_xg_against_home — synthetic xG of opponent (aimed AT home team)
  8. advanced_xg_against_away — synthetic xG of opponent (aimed AT away team)
  9. h2h_avg_goals         — average total goals in last 10 head-to-head meetings
 10. league_avg_goals      — running average goals per match in this league

All computations are strictly temporal: each row only uses data from matches
that occurred BEFORE that row's date.

Usage:
    python scripts/enrich_features.py [--data data/features_full.csv] [--window 5]
"""
import sys
from pathlib import Path
import argparse
import logging
import shutil

import numpy as np
import pandas as pd
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)

# Synthetic xG coefficients (match SyntheticXGBuilder and AdvancedFeatureBuilder)
SYNTH_XG_COEFS = {"shots": 0.03, "sot": 0.12, "intercept": 0.05}


def compute_synth_xg(shots: float, sot: float) -> float:
    """Synthetic xG from shot data."""
    return max(0.0, SYNTH_XG_COEFS["shots"] * shots +
               SYNTH_XG_COEFS["sot"] * sot +
               SYNTH_XG_COEFS["intercept"])


def enrich(df: pd.DataFrame, window: int = 5, h2h_window: int = 10) -> pd.DataFrame:
    """
    Add all missing features to the DataFrame.

    IMPORTANT: df must be sorted by Date before calling this function.
    All computations are strictly backward-looking to prevent data leakage.
    """
    n = len(df)
    logger.info(f"Enriching {n} matches with window={window}, h2h_window={h2h_window}")

    # Pre-allocate output columns
    form_home_ga = np.full(n, np.nan)
    form_away_ga = np.full(n, np.nan)
    attack_strength_home = np.full(n, np.nan)
    attack_strength_away = np.full(n, np.nan)
    defense_strength_home = np.full(n, np.nan)
    defense_strength_away = np.full(n, np.nan)
    adv_xg_against_home = np.full(n, np.nan)
    adv_xg_against_away = np.full(n, np.nan)
    h2h_avg_goals_arr = np.full(n, np.nan)
    league_avg_goals_arr = np.full(n, np.nan)

    # Tracking structures (all backward-looking)
    # Team goal history: team -> deque of (goals_scored, goals_conceded)
    team_goals_scored: dict[str, list[float]] = defaultdict(list)   # goals scored
    team_goals_conceded: dict[str, list[float]] = defaultdict(list)  # goals conceded
    team_shots_against: dict[str, list[tuple[float, float]]] = defaultdict(list)  # (shots, sot) AGAINST team

    # H2H: frozenset(team1, team2) -> list of total_goals
    h2h_history: dict[frozenset, list[float]] = defaultdict(list)

    # League running totals: league -> (total_goals, total_matches)
    league_totals: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    home_teams = df["HomeTeam"].values
    away_teams = df["AwayTeam"].values
    fthg = df["FTHG"].values.astype(float)
    ftag = df["FTAG"].values.astype(float)
    leagues = df["League"].values

    # Use HS/AS/HST/AST for synthetic xG if available
    has_shots = "HS" in df.columns and "AS" in df.columns
    has_sot = "HST" in df.columns and "AST" in df.columns

    if has_shots:
        hs = df["HS"].fillna(0).values.astype(float)
        as_ = df["AS"].fillna(0).values.astype(float)
    else:
        hs = np.zeros(n)
        as_ = np.zeros(n)

    if has_sot:
        hst = df["HST"].fillna(0).values.astype(float)
        ast = df["AST"].fillna(0).values.astype(float)
    else:
        hst = np.zeros(n)
        ast = np.zeros(n)

    logger.info(f"Shot data available: HS/AS={has_shots}, HST/AST={has_sot}")

    for i in range(n):
        ht = home_teams[i]
        at = away_teams[i]
        lg = leagues[i]
        home_goals = fthg[i]
        away_goals = ftag[i]

        # Skip if goals are NaN
        if np.isnan(home_goals) or np.isnan(away_goals):
            continue

        # ─── COMPUTE FEATURES FOR THIS ROW (using ONLY past data) ───

        # 1. form_home_ga: avg goals conceded by home team in last `window` matches
        if len(team_goals_conceded[ht]) > 0:
            recent = team_goals_conceded[ht][-window:]
            form_home_ga[i] = np.mean(recent)
        else:
            form_home_ga[i] = 0.0

        # 2. form_away_ga: avg goals conceded by away team in last `window` matches
        if len(team_goals_conceded[at]) > 0:
            recent = team_goals_conceded[at][-window:]
            form_away_ga[i] = np.mean(recent)
        else:
            form_away_ga[i] = 0.0

        # 3-6. attack/defense strength relative to league average
        lg_total, lg_count = league_totals[lg]
        if lg_count >= 10:  # need minimum sample for league avg
            lg_avg = lg_total / lg_count  # avg goals per TEAM per match ≈ 1.3-1.5
        else:
            lg_avg = 1.37  # fallback

        league_avg_goals_arr[i] = round(lg_avg, 3)

        # Attack: team's avg goals scored / league avg
        if len(team_goals_scored[ht]) >= 3:
            ht_gs = np.mean(team_goals_scored[ht][-window:])
            attack_strength_home[i] = round(ht_gs / max(lg_avg, 0.1), 3)
        else:
            attack_strength_home[i] = 1.0  # neutral

        if len(team_goals_scored[at]) >= 3:
            at_gs = np.mean(team_goals_scored[at][-window:])
            attack_strength_away[i] = round(at_gs / max(lg_avg, 0.1), 3)
        else:
            attack_strength_away[i] = 1.0

        # Defense: team's avg goals conceded / league avg
        if len(team_goals_conceded[ht]) >= 3:
            ht_gc = np.mean(team_goals_conceded[ht][-window:])
            defense_strength_home[i] = round(ht_gc / max(lg_avg, 0.1), 3)
        else:
            defense_strength_home[i] = 1.0

        if len(team_goals_conceded[at]) >= 3:
            at_gc = np.mean(team_goals_conceded[at][-window:])
            defense_strength_away[i] = round(at_gc / max(lg_avg, 0.1), 3)
        else:
            defense_strength_away[i] = 1.0

        # 7-8. Advanced xG against (synthetic xG of opponent's shots AGAINST this team)
        if len(team_shots_against[ht]) > 0:
            recent_against = team_shots_against[ht][-window:]
            adv_xg_against_home[i] = round(
                np.mean([compute_synth_xg(s, sot) for s, sot in recent_against]), 3
            )
        else:
            adv_xg_against_home[i] = 0.0

        if len(team_shots_against[at]) > 0:
            recent_against = team_shots_against[at][-window:]
            adv_xg_against_away[i] = round(
                np.mean([compute_synth_xg(s, sot) for s, sot in recent_against]), 3
            )
        else:
            adv_xg_against_away[i] = 0.0

        # 9. H2H avg goals
        matchup = frozenset([ht, at])
        if len(h2h_history[matchup]) > 0:
            recent_h2h = h2h_history[matchup][-h2h_window:]
            h2h_avg_goals_arr[i] = round(np.mean(recent_h2h), 3)
        else:
            h2h_avg_goals_arr[i] = round(lg_avg * 2, 3)  # fallback: league expected total

        # ─── UPDATE TRACKING STRUCTURES (after computing features) ───
        # This ensures we never use current-match data for current-match features

        team_goals_scored[ht].append(home_goals)
        team_goals_scored[at].append(away_goals)
        team_goals_conceded[ht].append(away_goals)  # home team concedes away_goals
        team_goals_conceded[at].append(home_goals)  # away team concedes home_goals

        # Shots against: home team faces away team's shots
        team_shots_against[ht].append((as_[i], ast[i]))
        team_shots_against[at].append((hs[i], hst[i]))

        # H2H
        h2h_history[matchup].append(home_goals + away_goals)

        # League totals (each match contributes 2 team-matches worth of goals)
        old_total, old_count = league_totals[lg]
        # League avg goals is per-team-per-match, so total_goals / (2 * matches)
        # But simpler: track per-match total and divide by match count
        league_totals[lg] = (old_total + (home_goals + away_goals) / 2.0, old_count + 1)

        if (i + 1) % 5000 == 0:
            logger.info(f"  Processed {i + 1}/{n} matches ({100*(i+1)/n:.0f}%)")

    # Assign to DataFrame
    df["form_home_ga"] = form_home_ga
    df["form_away_ga"] = form_away_ga
    df["attack_strength_home"] = attack_strength_home
    df["attack_strength_away"] = attack_strength_away
    df["defense_strength_home"] = defense_strength_home
    df["defense_strength_away"] = defense_strength_away
    df["advanced_xg_against_home"] = adv_xg_against_home
    df["advanced_xg_against_away"] = adv_xg_against_away
    df["h2h_avg_goals"] = h2h_avg_goals_arr
    df["league_avg_goals"] = league_avg_goals_arr

    return df


def validate(df: pd.DataFrame) -> bool:
    """Validate enriched DataFrame."""
    new_cols = [
        "form_home_ga", "form_away_ga",
        "attack_strength_home", "attack_strength_away",
        "defense_strength_home", "defense_strength_away",
        "advanced_xg_against_home", "advanced_xg_against_away",
        "h2h_avg_goals", "league_avg_goals",
    ]

    all_ok = True
    logger.info("\n=== VALIDATION ===")
    for col in new_cols:
        if col not in df.columns:
            logger.error(f"  ❌ {col} — MISSING from DataFrame")
            all_ok = False
            continue

        na_count = df[col].isna().sum()
        na_pct = na_count / len(df) * 100
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()

        status = "✅" if na_pct < 5 else "⚠️"
        logger.info(
            f"  {status} {col:30s} | "
            f"NaN: {na_count:>5d} ({na_pct:>5.1f}%) | "
            f"Mean: {mean_val:>7.3f} | Std: {std_val:>7.3f} | "
            f"Range: [{min_val:.3f}, {max_val:.3f}]"
        )

        # Sanity checks
        if col.startswith("attack_strength") or col.startswith("defense_strength"):
            if mean_val < 0.3 or mean_val > 3.0:
                logger.warning(f"    ⚠️ {col} mean looks suspicious: {mean_val:.3f}")
                all_ok = False
        elif col == "league_avg_goals":
            if mean_val < 0.5 or mean_val > 3.0:
                logger.warning(f"    ⚠️ {col} mean looks suspicious: {mean_val:.3f}")
                all_ok = False

    # Also verify BTTS features are now complete
    BTTS_FEATURES = [
        "form_home_gf", "form_home_ga",
        "form_away_gf", "form_away_ga",
        "home_clean_sheet_pct", "away_clean_sheet_pct",
        "home_scored_pct", "away_scored_pct",
        "synth_xg_home", "advanced_xg_against_home",
        "synth_xg_away", "advanced_xg_against_away",
        "defense_strength_home", "defense_strength_away",
        "attack_strength_home", "attack_strength_away",
        "elo_home", "elo_away",
        "h2h_avg_goals",
        "league_avg_goals",
    ]

    # Note: clean_sheet_pct and scored_pct are auto-computed by BTTSModel._add_computed_features
    # They need form_home_ga/form_away_ga which we just generated
    auto_computed = {"home_clean_sheet_pct", "away_clean_sheet_pct",
                     "home_scored_pct", "away_scored_pct"}

    logger.info("\n=== BTTS FEATURE COVERAGE ===")
    found = 0
    for feat in BTTS_FEATURES:
        if feat in df.columns:
            found += 1
            logger.info(f"  ✅ {feat}")
        elif feat in auto_computed:
            found += 1
            logger.info(f"  🔧 {feat} (auto-computed at training time)")
        else:
            logger.error(f"  ❌ {feat} — STILL MISSING")
            all_ok = False

    logger.info(f"\nBTTS coverage: {found}/{len(BTTS_FEATURES)} features")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Enrich features CSV")
    parser.add_argument("--data", type=str, default="data/features_full.csv")
    parser.add_argument("--window", type=int, default=5, help="Rolling window size")
    parser.add_argument("--h2h-window", type=int, default=10, help="H2H history window")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup")
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    # 1. Load
    logger.info(f"Loading {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    original_cols = set(df.columns)
    original_len = len(df)
    logger.info(f"Loaded {original_len} rows, {len(original_cols)} columns")

    # 2. Sort by date (critical for temporal correctness)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=False, errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
        logger.info(f"Sorted by Date: {df['Date'].min()} → {df['Date'].max()}")
    else:
        logger.warning("No Date column — temporal order not guaranteed!")

    # 3. Backup
    if not args.no_backup:
        backup_path = data_path.with_suffix(".csv.bak")
        if not backup_path.exists():
            shutil.copy2(data_path, backup_path)
            logger.info(f"Backup saved to {backup_path}")
        else:
            logger.info(f"Backup already exists at {backup_path}")

    # 4. Enrich
    df = enrich(df, window=args.window, h2h_window=args.h2h_window)

    # 5. Validate
    valid = validate(df)
    if not valid:
        logger.warning("Validation had warnings — review output above")

    # 6. Verify no data loss
    assert len(df) == original_len, f"Row count changed: {original_len} → {len(df)}"
    for col in original_cols:
        assert col in df.columns, f"Original column {col} lost!"
    logger.info(f"\n✅ No data loss: {original_len} rows, {len(original_cols)} original columns preserved")

    # 7. Save
    # Convert Date back to string for CSV compatibility
    if "Date" in df.columns:
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df.to_csv(data_path, index=False)
    new_cols = set(df.columns) - original_cols
    logger.info(f"✅ Saved enriched CSV: {len(df.columns)} columns (+{len(new_cols)} new)")
    logger.info(f"   New columns: {sorted(new_cols)}")


if __name__ == "__main__":
    main()
