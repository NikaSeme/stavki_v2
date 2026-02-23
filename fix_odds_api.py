import csv
from pathlib import Path
from difflib import SequenceMatcher
from stavki.data.collectors.odds_api import OddsAPIClient
from stavki.data.processors.normalize import mapper, _basic_normalize
from stavki.data.schemas import League

ODDS_CSV_PATH = Path("data/mapping/sources/odds_api.csv")

def best_match(raw_name, canonicals):
    # Try direct normalized subset match
    norm_raw = _basic_normalize(raw_name)
    
    # Strip common suffixes for matching logic
    suffixes = [" fc", " bc", " afc", " rovers", " wanderers", " city", " town", " united", " athletic", " albion", " county"]
    stripped_raw = norm_raw
    for s in suffixes:
        if stripped_raw.endswith(s):
            stripped_raw = stripped_raw[:-len(s)].strip()
            break
            
    best = None
    best_score = 0
    
    for c in canonicals:
        c_norm = _basic_normalize(c)
        if stripped_raw == c_norm:
            return c
            
        score = SequenceMatcher(None, stripped_raw, c_norm).ratio()
        if score > best_score:
            best_score = score
            best = c
            
    if best_score > 0.70:
        return best
    return None

def main():
    client = OddsAPIClient()
    canonicals = mapper.canonical_teams
    
    # Load existing to avoid duplicates
    existing = set()
    if ODDS_CSV_PATH.exists():
        with open(ODDS_CSV_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["raw_name"])
                
    new_mappings = []

    print("Fetching Odds API teams...")
    for league, sport in OddsAPIClient.SPORT_KEYS.items():
        if league == League.NBA: continue
        resp = client.get_odds(sport)
        if not resp.success or not resp.data: continue
        
        for match in resp.data:
            for raw in [match["home_team"], match["away_team"]]:
                if raw in existing: continue
                # Is it already mapped cleanly?
                mapped = mapper.map_name(raw)
                if not mapped:
                    # Let's find it!
                    suggestion = best_match(raw, canonicals)
                    if suggestion:
                        print(f"✅ Mapped: {raw} -> {suggestion}")
                        new_mappings.append({"raw_name": raw, "canonical_name": suggestion})
                        existing.add(raw)
                    else:
                        print(f"❌ Failed to find match for: {raw}")

    if new_mappings:
        write_header = not ODDS_CSV_PATH.exists()
        with open(ODDS_CSV_PATH, "a") as f:
            writer = csv.DictWriter(f, fieldnames=["raw_name", "canonical_name"])
            if write_header:
                writer.writeheader()
            for m in new_mappings:
                writer.writerow(m)
        print(f"Saved {len(new_mappings)} new mappings to {ODDS_CSV_PATH}!")
    else:
        print("No new mappings to save.")

if __name__ == "__main__":
    main()
