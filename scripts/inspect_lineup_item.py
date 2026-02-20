
import sys
import json
import gzip
from pathlib import Path

def inspect_lineup_keys(path):
    print(f"🕵️♂️ Inspecting Lineup Item in {path}")
    with gzip.open(path, 'rt', encoding='UTF-8') as f:
        data = json.load(f)
    
    lineups = data.get('lineups', [])
    if lineups:
        item = lineups[0]
        print(f"\nKeys in Lineup Item:")
        for k, v in item.items():
            if k == 'details':
                print(f"   - {k}: (List of {len(v)})")
            elif k == 'player':
                print(f"   - {k}: (Player Object)")
            else:
                print(f"   - {k}: {v}")
                
        # Also check a few items to see variation
        print("\nFormation Position Check:")
        for i, p in enumerate(lineups[:15]):
            print(f"   {i}. PosID: {p.get('position_id')}, FormPos: {p.get('formation_position')}, TypeID: {p.get('type_id')}")

if __name__ == "__main__":
    # Find a file
    import glob
    files = glob.glob("data/raw/fixtures/*/*.json.gz")
    if files:
        files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        inspect_lineup_keys(files[0])
