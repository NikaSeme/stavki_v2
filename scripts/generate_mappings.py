
import sys
import json
import csv
from pathlib import Path
from difflib import SequenceMatcher
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Use existing normalizer to get raw -> normalized
from stavki.data.processors.normalize import normalize_team_name

DATA_DIR = PROJECT_ROOT / "data"
MAPPING_DIR = DATA_DIR / "mapping"
SOURCES_DIR = MAPPING_DIR / "sources"
SOURCES_DIR.mkdir(exist_ok=True, parents=True)

def generate_mappings():
    print("🚀 Generating initial mapping CSVs...")
    
    # 1. Load Canonical Names
    try:
        df = pd.read_csv(DATA_DIR / "features_full.csv", usecols=["HomeTeam", "AwayTeam"], low_memory=False)
        canonical_names = set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique())
        canonical_names = {n for n in canonical_names if isinstance(n, str)}
    except Exception as e:
        print(f"❌ Failed to load features_full.csv: {e}")
        return

    # 2. SportMonks Mappings
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
    
    sm_mappings = []
    
    # Check manual overrides first (from normalize.py export if it exists)
    manual_overrides = {}
    overrides_path = MAPPING_DIR / "sportmonks_overrides.csv"
    if overrides_path.exists():
        with open(overrides_path, "r") as f:
             reader = csv.DictReader(f)
             for row in reader:
                 manual_overrides[row['raw_name']] = row['canonical_name']
    
    for raw in sorted(sm_teams):
        if not raw: continue
        
        # normalized = normalize_team_name(raw) # Don't use this, it's the OLD logic
        # We want to map Raw -> Canonical directly
        
        # 1. Exact match?
        if raw in canonical_names:
            sm_mappings.append((raw, raw))
            continue
            
        # 2. Manual override?
        if raw in manual_overrides:
             # Check if the override target is actually canonical
             target = manual_overrides[raw]
             # Normalize target to match canonical format (e.g., "Man Utd" -> "Man United"?)
             # Actually, let's just use what's in the override for now
             sm_mappings.append((raw, target))
             continue

        # 3. Fuzzy match against canonicals
        best_match = None
        best_score = 0.0
        
        # Simple normalization for matching
        raw_norm = raw.lower().replace(" fc", "").replace("cf ", "").strip()
        
        for canon in canonical_names:
            canon_norm = canon.lower().replace(" fc", "").replace("cf ", "").strip()
            
            # Exact match on normalized
            if raw_norm == canon_norm:
                best_match = canon
                best_score = 1.0
                break
                
            # Token set match (e.g. "Manchester City" vs "Man City")
            # "City" is in both
            
            # Use SequenceMatcher
            score = SequenceMatcher(None, raw_norm, canon_norm).ratio()
            if score > best_score:
                best_score = score
                best_match = canon
        
        if best_score > 0.6: # Pretty loose, but we want to suggest mappings
            sm_mappings.append((raw, best_match))
        else:
            sm_mappings.append((raw, "")) # Unmapped

    # Save SportMonks CSV
    sm_csv = SOURCES_DIR / "sportmonks.csv"
    with open(sm_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_name", "canonical_name"])
        for raw, canon in sm_mappings:
            writer.writerow([raw, canon])
            
    print(f"✅ Generated {sm_csv} with {len(sm_mappings)} rows")
    
    # 3. Odds API Stub
    # We don't have raw odds data easily accessible, so create a placeholder from overrides
    oa_csv = SOURCES_DIR / "odds_api.csv"
    with open(oa_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["raw_name", "canonical_name"])
        # Load from export if exists
        oa_overrides = MAPPING_DIR / "odds_api_overrides.csv"
        if oa_overrides.exists():
            with open(oa_overrides, "r") as fin:
                reader = csv.DictReader(fin)
                for row in reader:
                    writer.writerow([row['raw_name'], row['canonical_name']])
    
    print(f"✅ Generated {oa_csv} (placeholder)")

if __name__ == "__main__":
    generate_mappings()
