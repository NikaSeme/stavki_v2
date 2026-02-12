#!/usr/bin/env python3
"""
SportMonks Historical Backfill
==============================

Matches ~24K historical CSV rows to SportMonks fixture IDs,
then fetches xG / lineup stats.

Two phases:
  Phase 1: CSV row → fixture ID (via date + team name matching)
  Phase 2: fixture ID → xG stats (via get_fixture_stats)

Usage:
    # Phase 1 only (fast, ~2K calls)
    python3 scripts/backfill_sportmonks.py --phase 1

    # Phase 2 only (slow, ~24K calls)
    python3 scripts/backfill_sportmonks.py --phase 2

    # Both phases
    python3 scripts/backfill_sportmonks.py

    # Limit for testing
    python3 scripts/backfill_sportmonks.py --phase 1 --limit 10
"""

import sys
import logging
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backfill")

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

DATA_DIR = PROJECT_ROOT / "data"
FIXTURE_MAP_PATH = DATA_DIR / "fixture_id_map.json"
ENRICHED_PATH = DATA_DIR / "features_enriched.parquet"
FEATURES_CSV = DATA_DIR / "features_full.csv"

# football-data.co.uk Div → SportMonks league_id
LEAGUE_MAP = {
    "E0": 8,    # Premier League
    "E1": 9,    # Championship
    "D1": 82,   # Bundesliga
    "I1": 384,  # Serie A
    "SP1": 564, # La Liga
    "F1": 301,  # Ligue 1
}

# Hardcoded team name mappings: CSV name → possible SportMonks names
# This handles abbreviations and special characters that fuzzy matching struggles with.
TEAM_NAME_MAP = {
    # EPL / Championship
    "man city": "manchester city",
    "man united": "manchester united",
    "nott'm forest": "nottingham forest",
    "newcastle": "newcastle united",
    "brighton": "brighton and hove albion",
    "leicester": "leicester city",
    "west ham": "west ham united",
    "west brom": "west bromwich albion",
    "wolves": "wolverhampton wanderers",
    "tottenham": "tottenham hotspur",
    "crystal palace": "crystal palace",
    "sheffield united": "sheffield united",
    "sheffield weds": "sheffield wednesday",
    "norwich": "norwich city",
    "stoke": "stoke city",
    "cardiff": "cardiff city",
    "hull": "hull city",
    "swansea": "swansea city",
    "luton": "luton town",
    "burnley": "burnley",
    "bournemouth": "afc bournemouth",
    "leeds": "leeds united",
    "huddersfield": "huddersfield town",
    "watford": "watford",
    "fulham": "fulham",
    "sunderland": "sunderland",
    "middlesbrough": "middlesbrough",
    "ipswich": "ipswich town",
    "coventry": "coventry city",
    "derby": "derby county",
    "reading": "reading",
    "bristol city": "bristol city",
    "blackburn": "blackburn rovers",
    "blackpool": "blackpool",
    "bolton": "bolton wanderers",
    "millwall": "millwall",
    "preston": "preston north end",
    "rotherham": "rotherham united",
    "peterboro": "peterborough united",
    "plymouth": "plymouth argyle",
    "wigan": "wigan athletic",
    "barnsley": "barnsley",
    "charlton": "charlton athletic",
    "birmingham": "birmingham city",
    "qpr": "queens park rangers",
    "burton": "burton albion",
    "wycombe": "wycombe wanderers",
    "milton keynes dons": "mk dons",  
    "brentford": "brentford",
    "aston villa": "aston villa",
    "everton": "everton",
    "arsenal": "arsenal",
    "chelsea": "chelsea",
    "liverpool": "liverpool",
    "southampton": "southampton",
    
    # La Liga
    "ath madrid": "atletico de madrid",
    "ath bilbao": "athletic club",
    "sociedad": "real sociedad",
    "betis": "real betis",
    "celta": "celta de vigo",
    "espanol": "rcd espanyol",
    "vallecano": "rayo vallecano",
    "la coruna": "deportivo de la coruna",
    "sp gijon": "sporting de gijon",
    "alaves": "deportivo alaves",
    "valladolid": "real valladolid",
    "mallorca": "rcd mallorca",
    "cadiz": "cadiz",
    "las palmas": "ud las palmas",
    "leganes": "cd leganes",
    "huesca": "sd huesca",
    "cordoba": "cordoba",
    "eibar": "sd eibar",
    "elche": "elche",
    "getafe": "getafe",
    "girona": "girona",
    "granada": "granada",
    "malaga": "malaga",
    "osasuna": "ca osasuna",
    "almeria": "ud almeria",
    "barcelona": "fc barcelona",
    "real madrid": "real madrid",
    "sevilla": "sevilla",
    "valencia": "valencia",
    "villarreal": "villarreal",
    
    # Bundesliga
    "m'gladbach": "borussia monchengladbach",
    "dortmund": "borussia dortmund",
    "ein frankfurt": "eintracht frankfurt",
    "leverkusen": "bayer 04 leverkusen",
    "bayern munich": "fc bayern munchen",
    "fc koln": "1. fc koln",
    "rb leipzig": "rb leipzig",
    "schalke 04": "fc schalke 04",
    "hertha": "hertha bsc",
    "werder bremen": "werder bremen",
    "wolfsburg": "vfl wolfsburg",
    "augsburg": "fc augsburg",
    "freiburg": "sc freiburg",
    "hoffenheim": "tsg hoffenheim",
    "mainz": "fsv mainz 05",
    "stuttgart": "vfb stuttgart",
    "union berlin": "fc union berlin",
    "bochum": "vfl bochum 1848",
    "ingolstadt": "fc ingolstadt 04",
    "paderborn": "sc paderborn 07",
    "darmstadt": "darmstadt 98",
    "hamburg": "hamburger sv",
    "hannover": "hannover 96",
    "nurnberg": "1. fc nurnberg",
    "fortuna dusseldorf": "fortuna dusseldorf",
    "greuther furth": "greuther furth",
    "bielefeld": "arminia bielefeld",
    "heidenheim": "1. fc heidenheim 1846",
    
    # Serie A
    "inter": "internazionale",
    "milan": "ac milan",
    "roma": "as roma",
    "napoli": "ssc napoli",
    "lazio": "ss lazio",
    "juventus": "juventus",
    "atalanta": "atalanta",
    "fiorentina": "acf fiorentina",
    "torino": "torino",
    "sampdoria": "uc sampdoria",
    "genoa": "genoa",
    "bologna": "bologna",
    "udinese": "udinese",
    "sassuolo": "us sassuolo",
    "cagliari": "cagliari",
    "parma": "parma calcio 1913",
    "verona": "hellas verona",
    "spal": "spal 2013",
    "lecce": "us lecce",
    "brescia": "brescia",
    "crotone": "fc crotone",
    "benevento": "benevento",
    "spezia": "spezia calcio",
    "salernitana": "us salernitana 1919",
    "empoli": "empoli",
    "venezia": "venezia",
    "monza": "ac monza",
    "frosinone": "frosinone calcio",
    "cremonese": "us cremonese",
    "chievo": "chievo verona",
    "pescara": "pescara",
    "palermo": "palermo",
    "carpi": "carpi",
    "cesena": "cesena",

    # Ligue 1
    "paris sg": "paris saint-germain",
    "st etienne": "as saint-etienne",
    "marseille": "olympique de marseille",
    "lyon": "olympique lyonnais",
    "monaco": "as monaco",
    "lille": "losc lille",
    "nice": "ogc nice",
    "rennes": "stade rennais",
    "montpellier": "montpellier hsc",
    "nantes": "fc nantes",
    "bordeaux": "girondins de bordeaux",
    "toulouse": "toulouse",
    "strasbourg": "rc strasbourg alsace",
    "reims": "stade de reims",
    "lens": "rc lens",
    "brest": "stade brestois 29",
    "angers": "angers sco",
    "lorient": "fc lorient",
    "metz": "fc metz",
    "dijon": "dijon fco",
    "caen": "sm caen",
    "amiens": "amiens sc",
    "guingamp": "ea guingamp",
    "nimes": "nimes olympique",
    "troyes": "estac troyes",
    "clermont": "clermont foot 63",
    "bastia": "sc bastia",
    "le havre": "le havre ac",
    "auxerre": "aj auxerre",
    "nancy": "as nancy-lorraine",
    "ajaccio": "ac ajaccio",
    "ajaccio gfco": "gfc ajaccio",
    "evian thonon gaillard": "evian thonon gaillard",
}


def normalize(name: str) -> str:
    """Normalize a team name for comparison."""
    import unicodedata
    # Remove accents
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    # Lowercase
    name = name.lower().strip()
    # Remove common suffixes/noise
    for suffix in [" fc", " cf", " sc", " ac"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
    return name


def fuzzy_match(csv_name: str, api_names: list, threshold: float = 0.55) -> "Optional[str]":
    """
    Find the best fuzzy match for a CSV team name in a list of API names.
    Returns the API name or None if no match exceeds threshold.
    """
    csv_norm = normalize(csv_name)
    
    # 1. Check hardcoded map first
    if csv_norm in TEAM_NAME_MAP:
        mapped = normalize(TEAM_NAME_MAP[csv_norm])
        for api_name in api_names:
            if normalize(api_name) == mapped:
                return api_name
            # Also try partial match for the mapped name
            if mapped in normalize(api_name) or normalize(api_name) in mapped:
                return api_name
    
    # 2. Try exact normalized match
    for api_name in api_names:
        if normalize(api_name) == csv_norm:
            return api_name
    
    # 3. Try substring match (csv name inside API name or vice versa)
    for api_name in api_names:
        api_norm = normalize(api_name)
        if csv_norm in api_norm or api_norm in csv_norm:
            return api_name

    # 4. Fuzzy match as last resort
    best_score = 0.0
    best_match = None
    for api_name in api_names:
        score = SequenceMatcher(None, csv_norm, normalize(api_name)).ratio()
        if score > best_score:
            best_score = score
            best_match = api_name
    
    if best_score >= threshold:
        return best_match
    
    return None


# ─────────────────────────────────────────────────────────────
# Phase 1: Match CSV rows to fixture IDs
# ─────────────────────────────────────────────────────────────

def _swap_day_month(date_str: str) -> str:
    """Swap day and month in a YYYY-MM-DD string. Returns None if invalid."""
    try:
        parts = date_str.split("-")
        y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
        if d <= 12:  # Only ambiguous if day <= 12
            swapped = f"{y:04d}-{d:02d}-{m:02d}"
            # Validate it's a real date
            datetime.strptime(swapped, "%Y-%m-%d")
            return swapped
    except (ValueError, IndexError):
        pass
    return None


def _match_csv_rows_to_fixtures(csv_rows, fixtures, fixture_map, unmatched_log, date_str):
    """Match CSV rows to API fixtures, return count of matches found."""
    matched_count = 0
    still_unmatched = []
    
    for idx, row in csv_rows.iterrows():
        if str(idx) in fixture_map:
            matched_count += 1
            continue
            
        csv_home = row["HomeTeam"]
        csv_away = row["AwayTeam"]
        csv_div = row.get("Div", "")
        expected_league_id = LEAGUE_MAP.get(csv_div)
        
        best_fixture = None
        
        for fixture in fixtures:
            if expected_league_id and fixture.league_id != expected_league_id:
                continue
            
            home_match = fuzzy_match(csv_home, [fixture.home_team], threshold=0.45)
            away_match = fuzzy_match(csv_away, [fixture.away_team], threshold=0.45)
            
            if home_match and away_match:
                best_fixture = fixture
                break
        
        if best_fixture:
            fixture_map[str(idx)] = {
                "fixture_id": best_fixture.fixture_id,
                "csv_home": csv_home,
                "csv_away": csv_away,
                "api_home": best_fixture.home_team,
                "api_away": best_fixture.away_team,
                "date": date_str,
                "api_date": date_str,
                "league_id": best_fixture.league_id,
            }
            matched_count += 1
        else:
            still_unmatched.append({
                "date": date_str,
                "csv_home": csv_home,
                "csv_away": csv_away,
                "div": csv_div,
            })
    
    return matched_count, still_unmatched


def phase1_match_fixtures(client, df: pd.DataFrame, limit: int = 0) -> dict:
    """
    For each unique date in the CSV, fetch fixtures from SportMonks
    and match them to CSV rows by team name.
    
    Handles DD/MM vs MM/DD date ambiguity: when the original date yields
    no matches and day <= 12, tries the swapped date.
    
    Returns: dict mapping "csv_index" → fixture_id
    """
    # Load existing progress
    fixture_map = {}
    processed_dates = set()
    
    if FIXTURE_MAP_PATH.exists():
        with open(FIXTURE_MAP_PATH) as f:
            saved = json.load(f)
            fixture_map = saved.get("matches", {})
            processed_dates = set(saved.get("processed_dates", []))
        logger.info(f"Resuming: {len(fixture_map)} matches, {len(processed_dates)} dates done")
    
    # Group CSV rows by date
    df["_date_str"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    dates_groups = df.groupby("_date_str")
    
    all_dates = sorted(dates_groups.groups.keys())
    remaining = [d for d in all_dates if d not in processed_dates]
    logger.info(f"Total dates: {len(all_dates)}, remaining: {len(remaining)}")
    
    if limit > 0:
        remaining = remaining[:limit]
    
    our_league_ids = list(LEAGUE_MAP.values())
    unmatched_log = []
    date_swaps = 0
    
    for date_str in tqdm(remaining, desc="Phase 1: Matching fixtures"):
        try:
            csv_rows = dates_groups.get_group(date_str)
            total_rows = len(csv_rows)
            
            # --- Try original date ---
            fixtures = client.get_fixtures_by_date(date_str, league_ids=our_league_ids)
            matched, unmatched = _match_csv_rows_to_fixtures(
                csv_rows, fixtures, fixture_map, unmatched_log, date_str
            )
            
            # --- If poor match rate and date is ambiguous, try swapped ---
            swapped_date = _swap_day_month(date_str)
            if matched < total_rows and swapped_date and swapped_date != date_str:
                fixtures_swapped = client.get_fixtures_by_date(swapped_date, league_ids=our_league_ids)
                if fixtures_swapped:
                    matched2, unmatched2 = _match_csv_rows_to_fixtures(
                        csv_rows, fixtures_swapped, fixture_map, unmatched_log, swapped_date
                    )
                    if matched2 > matched:
                        date_swaps += 1
                        unmatched = unmatched2
                        matched = matched2
            
            # Record remaining unmatched
            unmatched_log.extend(unmatched)
            processed_dates.add(date_str)
            
            # Checkpoint every 20 dates
            if len(processed_dates) % 20 == 0:
                _save_fixture_map(fixture_map, processed_dates)
                
        except Exception as e:
            logger.error(f"Error processing date {date_str}: {e}")
            _save_fixture_map(fixture_map, processed_dates)
            continue
    
    # Final save
    _save_fixture_map(fixture_map, processed_dates)
    
    # Report
    total_csv = len(df)
    matched = len(fixture_map)
    logger.info(f"\n{'='*50}")
    logger.info(f"Phase 1 Complete")
    logger.info(f"  Matched: {matched}/{total_csv} ({100*matched/total_csv:.1f}%)")
    logger.info(f"  Unmatched: {len(unmatched_log)}")
    logger.info(f"  Date swaps needed: {date_swaps}")
    logger.info(f"  Dates processed: {len(processed_dates)}/{len(all_dates)}")
    
    if unmatched_log:
        unmatched_path = DATA_DIR / "unmatched_fixtures.json"
        with open(unmatched_path, "w") as f:
            json.dump(unmatched_log[:500], f, indent=2)
        logger.info(f"  Unmatched log saved to {unmatched_path}")
    
    return fixture_map


def _save_fixture_map(fixture_map: dict, processed_dates: set):
    """Save fixture map checkpoint."""
    with open(FIXTURE_MAP_PATH, "w") as f:
        json.dump({
            "matches": fixture_map,
            "processed_dates": sorted(processed_dates),
            "updated_at": datetime.now().isoformat(),
        }, f)
    logger.debug(f"Checkpoint: {len(fixture_map)} matches, {len(processed_dates)} dates")


# ─────────────────────────────────────────────────────────────
# Phase 2: Fetch xG stats for matched fixtures
# ─────────────────────────────────────────────────────────────

def phase2_fetch_stats(client, fixture_map: dict, limit: int = 0) -> pd.DataFrame:
    """
    For each matched fixture, fetch FULL data from SportMonks:
    - xG and match statistics (shots, possession, corners, fouls, cards)
    - Lineups with formations and coach names
    - Match events (goals, cards, substitutions)
    - Referee information
    
    Uses ONE API call per fixture via get_fixture_full().
    Saves results incrementally to parquet every 100 fixtures.
    """
    # Load existing enriched data
    if ENRICHED_PATH.exists():
        enriched_df = pd.read_parquet(ENRICHED_PATH)
        fetched_ids = set(str(x) for x in enriched_df["fixture_id"].dropna().unique())
        logger.info(f"Resuming: {len(fetched_ids)} fixtures already fetched")
    else:
        enriched_df = pd.DataFrame()
        fetched_ids = set()
    
    # Find fixtures needing stats
    to_fetch = []
    for csv_idx, info in fixture_map.items():
        fid = str(info["fixture_id"])
        if fid not in fetched_ids:
            to_fetch.append((csv_idx, info))
    
    logger.info(f"Fixtures to fetch: {len(to_fetch)}")
    
    if limit > 0:
        to_fetch = to_fetch[:limit]
    
    new_rows = []
    errors = 0
    stats_count = 0
    lineup_count = 0
    referee_count = 0
    
    for csv_idx, info in tqdm(to_fetch, desc="Phase 2: Fetching full data"):
        fid = info["fixture_id"]
        
        try:
            full = client.get_fixture_full(fid)
            stats = full.get("stats")
            lineups = full.get("lineups", {})
            events = full.get("events", [])
            referee = full.get("referee")
            
            row = {
                "csv_index": int(csv_idx),
                "fixture_id": fid,
                "date": info["date"],
                "csv_home": info["csv_home"],
                "csv_away": info["csv_away"],
                "api_home": info["api_home"],
                "api_away": info["api_away"],
                "league_id": info["league_id"],
            }
            
            # --- Core statistics ---
            if stats:
                stats_count += 1
                row["xg_home"] = stats.home_xg
                row["xg_away"] = stats.away_xg
                row["shots_home"] = stats.home_shots
                row["shots_away"] = stats.away_shots
                row["sot_home"] = stats.home_shots_on_target
                row["sot_away"] = stats.away_shots_on_target
                row["possession_home"] = stats.home_possession
                row["possession_away"] = stats.away_possession
                row["corners_home"] = stats.home_corners
                row["corners_away"] = stats.away_corners
                row["fouls_home"] = stats.home_fouls
                row["fouls_away"] = stats.away_fouls
                row["yellow_cards_home"] = stats.home_yellow_cards
                row["yellow_cards_away"] = stats.away_yellow_cards
                row["red_cards_home"] = stats.home_red_cards
                row["red_cards_away"] = stats.away_red_cards
            else:
                row["xg_home"] = None
                row["xg_away"] = None
            
            # --- Referee ---
            if referee:
                referee_count += 1
                row["referee"] = referee
            else:
                row["referee"] = None
            
            # --- Lineups & formations ---
            home_lineup = lineups.get("home")
            away_lineup = lineups.get("away")
            
            if home_lineup or away_lineup:
                lineup_count += 1
            
            if home_lineup:
                row["formation_home"] = home_lineup.formation
                row["coach_home"] = home_lineup.coach
                row["lineup_home"] = json.dumps(
                    home_lineup.starting_xi, ensure_ascii=False
                )
            else:
                row["formation_home"] = None
                row["coach_home"] = None
                row["lineup_home"] = None
            
            if away_lineup:
                row["formation_away"] = away_lineup.formation
                row["coach_away"] = away_lineup.coach
                row["lineup_away"] = json.dumps(
                    away_lineup.starting_xi, ensure_ascii=False
                )
            else:
                row["formation_away"] = None
                row["coach_away"] = None
                row["lineup_away"] = None
            
            # --- Events summary (card counts from events as backup) ---
            goal_events = [e for e in events if "goal" in e.get("type", "")]
            card_events = [e for e in events if "card" in e.get("type", "")]
            row["event_count"] = len(events)
            row["goal_event_count"] = len(goal_events)
            row["card_event_count"] = len(card_events)
            
            row["fetched_at"] = datetime.now().isoformat()
            new_rows.append(row)
            
        except Exception as e:
            errors += 1
            logger.warning(f"Error fetching fixture {fid}: {e}")
            # Still record it so we don't retry
            new_rows.append({
                "csv_index": int(csv_idx),
                "fixture_id": fid,
                "date": info["date"],
                "csv_home": info["csv_home"],
                "csv_away": info["csv_away"],
                "error": str(e),
                "fetched_at": datetime.now().isoformat(),
            })
        
        # Checkpoint every 100 fixtures
        if len(new_rows) >= 100:
            enriched_df = _save_enriched(new_rows, enriched_df)
            logger.info(
                f"  Checkpoint: {len(enriched_df)} total | "
                f"stats: {stats_count} | lineups: {lineup_count} | "
                f"referees: {referee_count} | errors: {errors}"
            )
            new_rows = []
    
    # Final save
    if new_rows:
        enriched_df = _save_enriched(new_rows, enriched_df)
    
    # Report
    if not enriched_df.empty:
        has_xg = enriched_df["xg_home"].notna().sum() if "xg_home" in enriched_df.columns else 0
        has_lineup = (enriched_df["lineup_home"].str.len() > 2).sum() if "lineup_home" in enriched_df.columns else 0
        has_ref = enriched_df["referee"].notna().sum() if "referee" in enriched_df.columns else 0
        total = len(enriched_df)
        logger.info(f"\n{'='*50}")
        logger.info(f"Phase 2 Complete")
        logger.info(f"  Total fetched: {total}")
        logger.info(f"  With xG data: {has_xg} ({100*has_xg/total:.1f}%)")
        logger.info(f"  With lineups: {has_lineup} ({100*has_lineup/total:.1f}%)")
        logger.info(f"  With referee: {has_ref} ({100*has_ref/total:.1f}%)")
        logger.info(f"  Errors: {errors}")
    
    return enriched_df


def _save_enriched(new_rows: list, existing_df: pd.DataFrame) -> pd.DataFrame:
    """Save enriched data checkpoint."""
    new_df = pd.DataFrame(new_rows)
    
    if not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    
    combined = combined.drop_duplicates(subset=["fixture_id"], keep="last")
    combined.to_parquet(ENRICHED_PATH, index=False)
    logger.debug(f"Checkpoint: {len(combined)} enriched rows")
    return combined


# ─────────────────────────────────────────────────────────────
# Phase 3: Merge enriched data back into features CSV
# ─────────────────────────────────────────────────────────────

def phase3_merge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge enriched data (xG, lineups, referee, events) back into 
    the main features DataFrame. Creates features_full_enriched.csv.
    """
    if not ENRICHED_PATH.exists():
        logger.error("No enriched data found. Run phase 2 first.")
        return df
    
    enriched = pd.read_parquet(ENRICHED_PATH)
    
    if enriched.empty:
        logger.warning("Enriched data is empty")
        return df
    
    # Create merge key: csv_index
    if "csv_index" not in enriched.columns:
        logger.error("Enriched data missing csv_index column")
        return df
    
    # Select columns to merge (all enrichment data)
    merge_cols = [
        "csv_index", "fixture_id",
        # Core stats
        "xg_home", "xg_away",
        "shots_home", "shots_away", "sot_home", "sot_away",
        "possession_home", "possession_away", "corners_home", "corners_away",
        # Card/foul stats
        "fouls_home", "fouls_away",
        "yellow_cards_home", "yellow_cards_away",
        "red_cards_home", "red_cards_away",
        # Referee
        "referee",
        # Lineups & formations
        "formation_home", "formation_away",
        "coach_home", "coach_away",
        "lineup_home", "lineup_away",
        # Events
        "event_count", "goal_event_count", "card_event_count",
    ]
    available = [c for c in merge_cols if c in enriched.columns]
    enriched_slim = enriched[available].copy()
    
    # Reset index to create merge key 
    df = df.reset_index(drop=True)
    df["csv_index"] = df.index
    
    # Merge
    merged = df.merge(enriched_slim, on="csv_index", how="left", suffixes=("", "_sm"))
    
    # Report
    has_xg = merged["xg_home"].notna().sum() if "xg_home" in merged.columns else 0
    has_lineup = (merged["lineup_home"].str.len() > 2).sum() if "lineup_home" in merged.columns else 0
    has_ref = merged["referee"].notna().sum() if "referee" in merged.columns else 0
    logger.info(f"Merged: {has_xg}/{len(merged)} with xG | {has_lineup} with lineups | {has_ref} with referee")
    
    # Save
    output_path = DATA_DIR / "features_full_enriched.csv"
    merged.to_csv(output_path, index=False)
    logger.info(f"Saved enriched features to {output_path}")
    
    return merged


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SportMonks historical backfill")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0,
                        help="Run specific phase (0=all)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit items to process (0=all)")
    args = parser.parse_args()
    
    # Load CSV
    logger.info(f"Loading {FEATURES_CSV}...")
    df = pd.read_csv(FEATURES_CSV, low_memory=False)
    logger.info(f"Loaded {len(df)} matches")
    
    # Init API client
    from stavki.config import get_config
    from stavki.data.collectors.sportmonks import SportMonksClient
    
    config = get_config()
    if not config.sportmonks_api_key:
        logger.error("SPORTMONKS_API_KEY not set!")
        sys.exit(1)
    
    client = SportMonksClient(api_key=config.sportmonks_api_key)
    
    run_all = args.phase == 0
    
    # Phase 1
    if run_all or args.phase == 1:
        logger.info("\n" + "="*50)
        logger.info("PHASE 1: Matching CSV rows to fixture IDs")
        logger.info("="*50)
        fixture_map = phase1_match_fixtures(client, df, limit=args.limit)
    
    # Phase 2
    if run_all or args.phase == 2:
        logger.info("\n" + "="*50)
        logger.info("PHASE 2: Fetching xG stats")
        logger.info("="*50)
        
        # Load fixture map
        if not FIXTURE_MAP_PATH.exists():
            logger.error("No fixture map found. Run phase 1 first.")
            sys.exit(1)
        with open(FIXTURE_MAP_PATH) as f:
            fixture_map = json.load(f).get("matches", {})
        
        phase2_fetch_stats(client, fixture_map, limit=args.limit)
    
    # Phase 3
    if run_all or args.phase == 3:
        logger.info("\n" + "="*50)
        logger.info("PHASE 3: Merging enriched data into features CSV")
        logger.info("="*50)
        phase3_merge(df)
    
    logger.info("\nDone! ✅")


if __name__ == "__main__":
    main()
