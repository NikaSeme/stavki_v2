
import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
import pandas as pd
from difflib import SequenceMatcher

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import existing normalizer for comparison
from stavki.data.processors.normalize import normalize_team_name, _basic_normalize, TEAM_ALIASES

DATA_DIR = PROJECT_ROOT / "data"
MAPPING_DIR = DATA_DIR / "mapping"
MAPPING_DIR.mkdir(exist_ok=True, parents=True)

def audit_teams():
    print("🔍 Starting Team Name Audit...")
    
    # 1. Load Canonical Names from features_full.csv
    # These are the names we train on. Any incoming name MUST map to one of these.
    features_path = DATA_DIR / "features_full.csv"
    if not features_path.exists():
        print("❌ features_full.csv not found!")
        return

    df = pd.read_csv(features_path, usecols=["HomeTeam", "AwayTeam", "League"], low_memory=False)
    canonical_names = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
    canonical_names = {n for n in canonical_names if isinstance(n, str)}
    
    print(f"✅ Found {len(canonical_names)} Canonical Teams in features_full.csv")
    
    # Save Canonical List
    with open(MAPPING_DIR / "canonical_teams.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["canonical_name", "league"])
        for name in sorted(canonical_names):
            league = df[df["HomeTeam"] == name]["League"].iloc[0] if not df[df["HomeTeam"] == name].empty else "Unknown"
            writer.writerow([name, league])

    # 1.5 Export Current Hardcoded Mappings
    from stavki.data.processors.normalize import TEAM_ALIASES, SourceNormalizer
    
    # Canonical Aliases
    with open(MAPPING_DIR / "current_aliases.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "canonical_name"])
        for alias, canon in sorted(TEAM_ALIASES.items()):
            writer.writerow([alias, canon])
            
    # Odds API Overrides
    with open(MAPPING_DIR / "odds_api_overrides.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_name", "canonical_name"])
        for raw, canon in sorted(SourceNormalizer.ODDS_API_OVERRIDES.items()):
            writer.writerow([raw, canon])
            
    # SportMonks Overrides
    with open(MAPPING_DIR / "sportmonks_overrides.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_name", "canonical_name"])
        for raw, canon in sorted(SourceNormalizer.SPORTMONKS_OVERRIDES.items()):
            writer.writerow([raw, canon])
            
    print("✅ Exported current hardcoded mappings to data/mapping/")

    # 2. Audit SportMonks Data
    # Load from sportmonks_historical.jsonl
    sm_path = PROJECT_ROOT / "stavki/data/storage/sportmonks_historical.jsonl"
    sm_teams = set()
    
    if sm_path.exists():
        with open(sm_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    participants = data.get("sm_data", {}).get("participants", [])
                    for p in participants:
                        sm_teams.add(p.get("name"))
                except:
                    pass
    
    print(f"✅ Found {len(sm_teams)} Distinct SportMonks Team Names")
    
    # Check mapping coverage
    mapping_issues = []
    mapped_count = 0
    
    for sm_name in sm_teams:
        if not sm_name: continue
        
        # Test current normalization
        normalized = normalize_team_name(sm_name)
        
        if normalized in canonical_names:
            mapped_count += 1
        else:
            # It didn't map to a Known Canonical Name
            # Is it because it normalized to something else?
            
            # Suggest a match
            best_match = None
            best_score = 0
            for canon in canonical_names:
                score = SequenceMatcher(None, normalized, canon).ratio()
                if score > best_score:
                    best_score = score
                    best_match = canon
            
            mapping_issues.append({
                "source": "SportMonks",
                "raw_name": sm_name,
                "normalized": normalized,
                "best_guess": best_match if best_score > 0.6 else None,
                "score": round(best_score, 3)
            })

    print(f"📊 SportMonks Coverage: {mapped_count}/{len(sm_teams)} ({mapped_count/len(sm_teams)*100:.1f}%)")
    
    if mapping_issues:
        print(f"⚠️  Found {len(mapping_issues)} unmapped teams. Saving to csv...")
        issue_df = pd.DataFrame(mapping_issues)
        issue_df.sort_values("score", ascending=False).to_csv(MAPPING_DIR / "sportmonks_mismatches.csv", index=False)
        print(f"   Saved to {MAPPING_DIR / 'sportmonks_mismatches.csv'}")
    
    # 3. Audit Odds/Bookmaker Data (if available)
    # Check raw directory for any recent Odds API dumps
    raw_dir = DATA_DIR / "raw"
    odds_teams = defaultdict(set) # bookmaker -> set of names
    
    # This is a best-effort check for JSON files in raw that might contain odds
    # (Assuming structure based on typical OddsAPI responses)
    # ... actually, without specific file paths, this might be hard.
    # Let's check typical 'upcoming_odds.json' locations if they exist
    # usage: stavki/data/collectors/odds_api.py saves to somewhere?
    
    # If explicit odds CSVs exist, use them.
    # Check for legacy odds files
    odds_files = list(Path("data/odds").glob("*.csv")) if Path("data/odds").exists() else []
    
    for details in mapping_issues:
         # Just print top 5 to console
         pass

if __name__ == "__main__":
    audit_teams()
