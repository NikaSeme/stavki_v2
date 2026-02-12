#!/usr/bin/env python3
"""
Compute Tier 1 Features
========================

Reads features_enriched.parquet (Phase 2 output) and computes:
  1. Synthetic xG from per-player shot data
  2. Player rating aggregates (avg XI rating, rating delta, key players)
  3. Referee profiles (goals/game, cards/game, home bias, strictness)

Then merges into features_full.csv for model training.

Usage:
    python3 scripts/compute_tier1_features.py
    python3 scripts/compute_tier1_features.py --merge   # Also update features_full.csv
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ENRICHED_PATH = DATA_DIR / "features_enriched.parquet"
FEATURES_CSV = DATA_DIR / "features_full.csv"
OUTPUT_PATH = DATA_DIR / "features_tier1.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tier1")


# =====================================================
# 1. Parse per-player data from lineup JSON
# =====================================================

def parse_lineup_json(lineup_json: str) -> list:
    """Parse lineup JSON string into list of player dicts."""
    if pd.isna(lineup_json) or not lineup_json:
        return []
    try:
        players = json.loads(lineup_json)
        if isinstance(players, list):
            return players
        return []
    except (json.JSONDecodeError, TypeError):
        return []


def extract_player_stats(players: list) -> dict:
    """
    Aggregate per-player stats for a team's starting XI.
    
    Returns dict with aggregated stats.
    """
    if not players:
        return {}
    
    ratings = []
    total_shots = 0
    total_sot = 0
    total_big_chances = 0
    total_big_chances_missed = 0
    total_key_passes = 0
    total_touches = 0
    total_minutes = 0
    key_player_count = 0  # Players rated >= 7.5
    player_count = len(players)
    
    for p in players:
        # Rating
        rating = p.get("rating")
        if rating is not None:
            try:
                r = float(rating)
                ratings.append(r)
                if r >= 7.5:
                    key_player_count += 1
            except (ValueError, TypeError):
                pass
        
        # Shots
        shots = p.get("shots")
        if shots is not None:
            try:
                total_shots += int(shots)
            except (ValueError, TypeError):
                pass
        
        sot = p.get("shots_on_target")
        if sot is not None:
            try:
                total_sot += int(sot)
            except (ValueError, TypeError):
                pass
        
        # Big chances
        bc = p.get("big_chances_created")
        if bc is not None:
            try:
                total_big_chances += int(bc)
            except (ValueError, TypeError):
                pass
        
        bcm = p.get("big_chances_missed")
        if bcm is not None:
            try:
                total_big_chances_missed += int(bcm)
            except (ValueError, TypeError):
                pass
        
        # Key passes
        kp = p.get("key_passes")
        if kp is not None:
            try:
                total_key_passes += int(kp)
            except (ValueError, TypeError):
                pass
        
        # Touches
        t = p.get("touches")
        if t is not None:
            try:
                total_touches += int(t)
            except (ValueError, TypeError):
                pass
        
        # Minutes
        m = p.get("minutes_played")
        if m is not None:
            try:
                total_minutes += int(m)
            except (ValueError, TypeError):
                pass
    
    result = {
        "player_count": player_count,
        "total_shots": total_shots,
        "total_sot": total_sot,
        "total_big_chances": total_big_chances,
        "total_big_chances_missed": total_big_chances_missed,
        "total_key_passes": total_key_passes,
        "total_touches": total_touches,
        "total_minutes": total_minutes,
        "key_player_count": key_player_count,
    }
    
    if ratings:
        result["avg_rating"] = np.mean(ratings)
        result["min_rating"] = min(ratings)
        result["max_rating"] = max(ratings)
        result["rating_std"] = np.std(ratings) if len(ratings) > 1 else 0.0
        result["has_ratings"] = True
    else:
        result["has_ratings"] = False
    
    return result


# =====================================================
# 2. Synthetic xG Model — trained on real goals data
# =====================================================

def calibrate_xg_model(df: pd.DataFrame) -> dict:
    """
    Train xG coefficients from historical data using actual goals as labels.
    
    Uses: shots, shots_on_target, big_chances → actual_goals
    Loads per-team goals from features_full.csv matched via csv_index.
    """
    # Load actual goals from the main CSV
    csv_path = DATA_DIR / "features_full.csv"
    if not csv_path.exists():
        logger.warning("features_full.csv not found for xG training. Using defaults.")
        return _default_xg_coefs()
    
    csv_df = pd.read_csv(csv_path, usecols=["FTHG", "FTAG"], low_memory=False)
    
    # Collect training data: (shots, sot, big_chances) → actual_goals
    X_data = []
    y_data = []
    
    for _, row in df.iterrows():
        csv_idx = row.get("csv_index")
        if pd.isna(csv_idx):
            continue
        csv_idx = int(csv_idx)
        
        # Get actual goals for this match from CSV
        if csv_idx >= len(csv_df):
            continue
        goals_home = csv_df.iloc[csv_idx].get("FTHG")
        goals_away = csv_df.iloc[csv_idx].get("FTAG")
        if pd.isna(goals_home) or pd.isna(goals_away):
            continue
        goals_home = float(goals_home)
        goals_away = float(goals_away)
        
        shots_h = row.get("shots_home")
        shots_a = row.get("shots_away")
        sot_h = row.get("sot_home")
        sot_a = row.get("sot_away")
        
        # Parse lineups for big chances
        home_players = parse_lineup_json(row.get("lineup_home"))
        away_players = parse_lineup_json(row.get("lineup_away"))
        home_stats = extract_player_stats(home_players)
        away_stats = extract_player_stats(away_players)
        
        bc_h = home_stats.get("total_big_chances", 0) + home_stats.get("total_big_chances_missed", 0)
        bc_a = away_stats.get("total_big_chances", 0) + away_stats.get("total_big_chances_missed", 0)
        
        # Home side training sample
        if pd.notna(shots_h) and pd.notna(sot_h):
            X_data.append([float(shots_h), float(sot_h), float(bc_h)])
            y_data.append(goals_home)
        elif pd.notna(sot_h):
            # SOT only (no total shots) — still useful
            X_data.append([float(sot_h) * 2.5, float(sot_h), float(bc_h)])
            y_data.append(goals_home)
        
        # Away side training sample
        if pd.notna(shots_a) and pd.notna(sot_a):
            X_data.append([float(shots_a), float(sot_a), float(bc_a)])
            y_data.append(goals_away)
        elif pd.notna(sot_a):
            X_data.append([float(sot_a) * 2.5, float(sot_a), float(bc_a)])
            y_data.append(goals_away)
    
    logger.info(f"xG training data: {len(X_data)} samples (shots+goals pairs)")
    
    if len(X_data) < 50:
        logger.warning(f"Not enough data ({len(X_data)} samples). Using defaults.")
        return _default_xg_coefs()
    
    # Train the regression: goals = a*shots + b*sot + c*big_chances + intercept
    X = np.array(X_data)
    y = np.array(y_data)
    
    model = LinearRegression()
    model.fit(X, y)
    
    coef_shots, coef_sot, coef_big_chances = model.coef_
    intercept = model.intercept_
    
    # Validate: coefficients should be positive (more shots → more goals)
    # If regression gives negative coefficients, clamp to small positive
    coef_shots = max(coef_shots, 0.005)
    coef_sot = max(coef_sot, 0.01)
    coef_big_chances = max(coef_big_chances, 0.05)
    intercept = max(intercept, 0.0)
    
    # Report model quality
    y_pred = X @ np.array([coef_shots, coef_sot, coef_big_chances]) + intercept
    r_squared = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    mae = np.mean(np.abs(y - y_pred))
    
    logger.info(f"xG model trained: R²={r_squared:.3f}, MAE={mae:.3f}")
    logger.info(f"  Coefficients: shots={coef_shots:.4f}, sot={coef_sot:.4f}, "
                f"big_chances={coef_big_chances:.4f}, intercept={intercept:.4f}")
    
    return {
        "coef_shots": round(coef_shots, 4),
        "coef_sot": round(coef_sot, 4),
        "coef_big_chances": round(coef_big_chances, 4),
        "intercept": round(intercept, 4),
        "calibrated": True,
        "r_squared": round(r_squared, 4),
        "mae": round(mae, 4),
        "n_samples": len(X_data),
    }


def _default_xg_coefs() -> dict:
    """Fallback xG coefficients when insufficient training data."""
    return {
        "coef_shots": 0.03,
        "coef_sot": 0.12,
        "coef_big_chances": 0.35,
        "intercept": 0.05,
        "calibrated": False,
    }


def compute_synth_xg(shots: float, sot: float, big_chances: float,
                     coefs: dict) -> float:
    """Compute synthetic xG from shot data using trained coefficients."""
    xg = (coefs["coef_shots"] * shots +
          coefs["coef_sot"] * sot +
          coefs["coef_big_chances"] * big_chances +
          coefs["intercept"])
    return max(0.0, round(xg, 3))


# =====================================================
# 3. Referee Profile Builder
# =====================================================

def build_referee_profiles(df: pd.DataFrame) -> dict:
    """
    Build per-referee profiles from historical data.
    
    Returns dict: referee_name → profile dict
    """
    profiles = defaultdict(lambda: {
        "matches": 0,
        "total_goals": 0,
        "total_cards": 0,
        "total_fouls": 0,
        "home_wins": 0,
        "draws": 0,
        "away_wins": 0,
        "over25_count": 0,
    })
    
    for _, row in df.iterrows():
        ref = row.get("referee")
        if pd.isna(ref) or not ref:
            continue
        
        ref = str(ref).strip().lower()
        p = profiles[ref]
        p["matches"] += 1
        
        # Goals
        goals = row.get("goal_event_count")
        if pd.notna(goals):
            g = int(goals)
            p["total_goals"] += g
            if g > 2:  # over 2.5 = 3+
                p["over25_count"] += 1
        
        # Cards (safe NaN handling)
        def safe_int(v):
            try:
                if pd.isna(v):
                    return 0
                return int(v)
            except (ValueError, TypeError):
                return 0
        
        yc_h = safe_int(row.get("yellow_cards_home"))
        yc_a = safe_int(row.get("yellow_cards_away"))
        rc_h = safe_int(row.get("red_cards_home"))
        rc_a = safe_int(row.get("red_cards_away"))
        p["total_cards"] += yc_h + yc_a + rc_h + rc_a
        
        # Fouls
        f_h = safe_int(row.get("fouls_home"))
        f_a = safe_int(row.get("fouls_away"))
        p["total_fouls"] += f_h + f_a
    
    # Compute averages
    result = {}
    for ref, p in profiles.items():
        n = p["matches"]
        if n < 3:  # Need minimum matches for reliable stats
            continue
        result[ref] = {
            "matches": n,
            "goals_per_game": round(p["total_goals"] / n, 2),
            "cards_per_game": round(p["total_cards"] / n, 2),
            "fouls_per_game": round(p["total_fouls"] / n, 2),
            "over25_rate": round(p["over25_count"] / n, 3),
            "experience": round(np.log1p(n), 2),
        }
    
    # Compute league averages for z-scores
    if result:
        avg_goals = np.mean([v["goals_per_game"] for v in result.values()])
        std_goals = np.std([v["goals_per_game"] for v in result.values()]) or 1.0
        avg_cards = np.mean([v["cards_per_game"] for v in result.values()])
        std_cards = np.std([v["cards_per_game"] for v in result.values()]) or 1.0
        
        for ref, p in result.items():
            p["goals_zscore"] = round((p["goals_per_game"] - avg_goals) / std_goals, 3)
            p["cards_zscore"] = round((p["cards_per_game"] - avg_cards) / std_cards, 3)
    
    logger.info(f"Built profiles for {len(result)} referees")
    return result


# =====================================================
# 4. Main Feature Computation
# =====================================================

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all Tier 1 features for each match.
    
    Uses ROLLING averages for predictive features (ratings, xG) — each match
    gets the team's pre-match profile, not post-match values.
    """
    logger.info(f"Computing Tier 1 features for {len(df)} matches...")
    
    # Load actual goals for per-team xG accuracy tracking
    csv_path = DATA_DIR / "features_full.csv"
    csv_goals = {}
    if csv_path.exists():
        csv_df = pd.read_csv(csv_path, usecols=["FTHG", "FTAG"], low_memory=False)
        for idx in range(len(csv_df)):
            row = csv_df.iloc[idx]
            if pd.notna(row.get("FTHG")):
                csv_goals[idx] = (float(row["FTHG"]), float(row["FTAG"]))
    
    # Calibrate xG model (trains on actual goals)
    xg_coefs = calibrate_xg_model(df)
    logger.info(f"xG coefficients: shots={xg_coefs['coef_shots']}, "
                f"sot={xg_coefs['coef_sot']}, big_chances={xg_coefs['coef_big_chances']}")
    
    # Build referee profiles  
    ref_profiles = build_referee_profiles(df)
    
    # Rolling state for teams (built chronologically)
    team_rating_history = defaultdict(list)  # team -> list of avg XI ratings
    team_xg_history = defaultdict(list)      # team -> list of synth_xg values
    team_goals_history = defaultdict(list)   # team -> list of actual goals
    ROLLING_WINDOW = 10  # Last N matches for rolling average
    
    # Sort by date for chronological rolling
    df = df.sort_values("date").reset_index(drop=True)
    
    # Defaults
    DEFAULT_RATING = 6.5
    DEFAULT_XG_HOME = 1.3
    DEFAULT_XG_AWAY = 1.1
    
    # Feature columns to fill
    new_cols = {
        "avg_rating_home": [], "avg_rating_away": [], "rating_delta": [],
        "key_players_home": [], "key_players_away": [],
        "xi_experience_home": [], "xi_experience_away": [],
        "synth_xg_home": [], "synth_xg_away": [], "synth_xg_diff": [],
        "ref_goals_per_game": [], "ref_cards_per_game_t1": [],
        "ref_over25_rate": [], "ref_strictness_t1": [],
        "ref_experience": [], "ref_goals_zscore": [],
    }
    
    for _, row in df.iterrows():
        home_team = str(row.get("csv_home", "")).strip()
        away_team = str(row.get("csv_away", "")).strip()
        csv_idx = row.get("csv_index")
        
        # --- Parse lineups ---
        home_players = parse_lineup_json(row.get("lineup_home"))
        away_players = parse_lineup_json(row.get("lineup_away"))
        home_stats = extract_player_stats(home_players)
        away_stats = extract_player_stats(away_players)
        
        # ============================================================
        # PLAYER RATINGS — use rolling team average (pre-match)
        # ============================================================
        
        # Get pre-match rolling rating for this team
        def get_rolling_rating(team, default):
            hist = team_rating_history.get(team, [])
            if not hist:
                return default
            recent = hist[-ROLLING_WINDOW:]
            return round(np.mean(recent), 2)
        
        rolling_r_h = get_rolling_rating(home_team, DEFAULT_RATING)
        rolling_r_a = get_rolling_rating(away_team, DEFAULT_RATING)
        
        # This match's actual XI rating (for updating rolling state)
        match_r_h = home_stats.get("avg_rating", None) if home_stats.get("has_ratings") else None
        match_r_a = away_stats.get("avg_rating", None) if away_stats.get("has_ratings") else None
        
        # Use rolling if available, current match rating if first appearance
        use_r_h = rolling_r_h if team_rating_history.get(home_team) else (match_r_h if match_r_h else DEFAULT_RATING)
        use_r_a = rolling_r_a if team_rating_history.get(away_team) else (match_r_a if match_r_a else DEFAULT_RATING)
        
        new_cols["avg_rating_home"].append(round(use_r_h, 2))
        new_cols["avg_rating_away"].append(round(use_r_a, 2))
        new_cols["rating_delta"].append(round(use_r_h - use_r_a, 2))
        new_cols["key_players_home"].append(home_stats.get("key_player_count", 0))
        new_cols["key_players_away"].append(away_stats.get("key_player_count", 0))
        
        # Experience: avg minutes per player in XI
        exp_h = home_stats.get("total_minutes", 0) / max(home_stats.get("player_count", 1), 1)
        exp_a = away_stats.get("total_minutes", 0) / max(away_stats.get("player_count", 1), 1)
        new_cols["xi_experience_home"].append(round(exp_h, 1))
        new_cols["xi_experience_away"].append(round(exp_a, 1))
        
        # Update rolling state AFTER using pre-match values
        if match_r_h is not None and home_team:
            team_rating_history[home_team].append(match_r_h)
        if match_r_a is not None and away_team:
            team_rating_history[away_team].append(match_r_a)
        
        # ============================================================
        # SYNTHETIC xG — use rolling team average (pre-match)
        # ============================================================
        
        def get_rolling_xg(team, default):
            hist = team_xg_history.get(team, [])
            if not hist:
                return default
            recent = hist[-ROLLING_WINDOW:]
            return round(np.mean(recent), 3)
        
        rolling_xg_h = get_rolling_xg(home_team, DEFAULT_XG_HOME)
        rolling_xg_a = get_rolling_xg(away_team, DEFAULT_XG_AWAY)
        
        # Compute this match's actual synth_xg for rolling state
        shots_h = home_stats.get("total_shots", 0)
        sot_h = home_stats.get("total_sot", 0)
        bc_h = home_stats.get("total_big_chances", 0) + home_stats.get("total_big_chances_missed", 0)
        shots_a = away_stats.get("total_shots", 0)
        sot_a = away_stats.get("total_sot", 0)
        bc_a = away_stats.get("total_big_chances", 0) + away_stats.get("total_big_chances_missed", 0)
        
        # Fall back to team-level shots
        if shots_h == 0 and pd.notna(row.get("shots_home")):
            shots_h = float(row["shots_home"])
            sot_h = float(row.get("sot_home", shots_h * 0.35))
        if shots_a == 0 and pd.notna(row.get("shots_away")):
            shots_a = float(row["shots_away"])
            sot_a = float(row.get("sot_away", shots_a * 0.35))
        
        match_xg_h = compute_synth_xg(shots_h, sot_h, bc_h, xg_coefs) if (shots_h > 0 or bc_h > 0) else None
        match_xg_a = compute_synth_xg(shots_a, sot_a, bc_a, xg_coefs) if (shots_a > 0 or bc_a > 0) else None
        
        # Use rolling for prediction, fall back to computed or default
        use_xg_h = rolling_xg_h if team_xg_history.get(home_team) else (match_xg_h if match_xg_h else DEFAULT_XG_HOME)
        use_xg_a = rolling_xg_a if team_xg_history.get(away_team) else (match_xg_a if match_xg_a else DEFAULT_XG_AWAY)
        
        new_cols["synth_xg_home"].append(use_xg_h)
        new_cols["synth_xg_away"].append(use_xg_a)
        new_cols["synth_xg_diff"].append(round(use_xg_h - use_xg_a, 3))
        
        # Update rolling state AFTER using pre-match values
        if match_xg_h is not None and home_team:
            team_xg_history[home_team].append(match_xg_h)
        if match_xg_a is not None and away_team:
            team_xg_history[away_team].append(match_xg_a)
        
        # ============================================================
        # REFEREE
        # ============================================================
        ref = row.get("referee")
        if pd.notna(ref) and str(ref).strip().lower() in ref_profiles:
            rp = ref_profiles[str(ref).strip().lower()]
            new_cols["ref_goals_per_game"].append(rp["goals_per_game"])
            new_cols["ref_cards_per_game_t1"].append(rp["cards_per_game"])
            new_cols["ref_over25_rate"].append(rp["over25_rate"])
            new_cols["ref_strictness_t1"].append(rp["cards_zscore"])
            new_cols["ref_experience"].append(rp["experience"])
            new_cols["ref_goals_zscore"].append(rp["goals_zscore"])
        else:
            new_cols["ref_goals_per_game"].append(2.7)
            new_cols["ref_cards_per_game_t1"].append(3.5)
            new_cols["ref_over25_rate"].append(0.55)
            new_cols["ref_strictness_t1"].append(0.0)
            new_cols["ref_experience"].append(0.0)
            new_cols["ref_goals_zscore"].append(0.0)
    
    # Add new columns to DataFrame
    for col, values in new_cols.items():
        df[col] = values
    
    # Report rolling stats
    teams_with_ratings = len([t for t, h in team_rating_history.items() if len(h) >= 3])
    teams_with_xg = len([t for t, h in team_xg_history.items() if len(h) >= 3])
    logger.info(f"Added {len(new_cols)} new feature columns")
    logger.info(f"Rolling stats: {teams_with_ratings} teams with 3+ rating samples, "
                f"{teams_with_xg} teams with 3+ xG samples")
    return df


# =====================================================
# 5. Merge into features_full.csv
# =====================================================

def merge_into_csv(tier1_df: pd.DataFrame, csv_path: Path) -> None:
    """
    Merge Tier 1 features into features_full.csv.
    
    Matches on fixture_id (enriched) ↔ csv_index (original row).
    """
    if not csv_path.exists():
        logger.error(f"CSV not found: {csv_path}")
        return
    
    logger.info(f"Loading {csv_path}...")
    df_csv = pd.read_csv(csv_path, low_memory=False)
    
    # The enriched parquet has csv_index that maps to the original CSV row index
    tier1_features = [
        "avg_rating_home", "avg_rating_away", "rating_delta",
        "key_players_home", "key_players_away",
        "xi_experience_home", "xi_experience_away",
        "synth_xg_home", "synth_xg_away", "synth_xg_diff",
        "ref_goals_per_game", "ref_cards_per_game_t1",
        "ref_over25_rate", "ref_strictness_t1",
        "ref_experience", "ref_goals_zscore",
    ]
    
    # Drop any existing Tier 1 columns to prevent duplicates on re-run
    existing_t1 = [c for c in tier1_features if c in df_csv.columns]
    if existing_t1:
        logger.info(f"Dropping {len(existing_t1)} existing Tier 1 columns from CSV")
        df_csv.drop(columns=existing_t1, inplace=True)
    original_cols = len(df_csv.columns)
    
    # Create merge key
    if "csv_index" not in tier1_df.columns:
        logger.error("No csv_index column for merging!")
        return
    
    merge_df = tier1_df[["csv_index"] + tier1_features].copy()
    merge_df["csv_index"] = merge_df["csv_index"].astype(int)
    
    # Merge on index
    df_csv["_row_idx"] = df_csv.index
    merged = df_csv.merge(
        merge_df,
        left_on="_row_idx",
        right_on="csv_index",
        how="left",
    )
    merged.drop(columns=["_row_idx", "csv_index"], inplace=True)
    
    # Fill NaN for unmatched rows with defaults
    defaults = {
        "avg_rating_home": 6.5, "avg_rating_away": 6.5, "rating_delta": 0.0,
        "key_players_home": 0, "key_players_away": 0,
        "xi_experience_home": 90.0, "xi_experience_away": 90.0,
        "synth_xg_home": 1.3, "synth_xg_away": 1.1, "synth_xg_diff": 0.2,
        "ref_goals_per_game": 2.7, "ref_cards_per_game_t1": 3.5,
        "ref_over25_rate": 0.55, "ref_strictness_t1": 0.0,
        "ref_experience": 0.0, "ref_goals_zscore": 0.0,
    }
    for col, default in defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)
    
    # Save
    output = csv_path.parent / "features_full_enriched.csv"
    merged.to_csv(output, index=False)
    
    matched = tier1_df["csv_index"].notna().sum()
    logger.info(f"Merged {matched}/{len(tier1_df)} matches into CSV")
    logger.info(f"Columns: {original_cols} → {len(merged.columns)} (+{len(merged.columns) - original_cols})")
    logger.info(f"Saved to {output}")
    logger.info(f"To use: cp {output} {csv_path}")


# =====================================================
# Main
# =====================================================

def main():
    parser = argparse.ArgumentParser(description="Compute Tier 1 features")
    parser.add_argument("--merge", action="store_true",
                        help="Also merge into features_full.csv")
    parser.add_argument("--input", default=str(ENRICHED_PATH),
                        help="Input enriched parquet path")
    parser.add_argument("--output", default=str(OUTPUT_PATH),
                        help="Output tier1 parquet path")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)
    
    # Load enriched data
    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} enriched matches")
    
    # Compute features
    df = compute_all_features(df)
    
    # Save tier1 parquet
    output_path = Path(args.output)
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    # Print sample
    tier1_cols = [c for c in df.columns if c.startswith(("avg_rating", "rating_delta",
                  "key_players", "synth_xg", "ref_", "xi_experience"))]
    print("\n📊 Sample Tier 1 Features (first 5 matches):")
    print(df[tier1_cols].head().to_string())
    
    # Print stats
    print(f"\n📈 Feature Statistics:")
    for col in tier1_cols:
        vals = df[col]
        non_default = vals[vals != vals.mode().iloc[0]] if len(vals.mode()) > 0 else vals
        print(f"  {col}: mean={vals.mean():.3f}, std={vals.std():.3f}, "
              f"non-default={len(non_default)}/{len(vals)}")
    
    # Merge into CSV if requested
    if args.merge:
        merge_into_csv(df, FEATURES_CSV)
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
