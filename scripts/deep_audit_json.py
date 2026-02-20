
import sys
import json
import gzip
from pathlib import Path
from collections import defaultdict

def recursive_keys(data, path="", keys_found=None):
    if keys_found is None:
        keys_found = defaultdict(set)
    
    if isinstance(data, dict):
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            keys_found[new_path].add(str(type(v).__name__))
            recursive_keys(v, new_path, keys_found)
    elif isinstance(data, list):
        if data:
            # Analyze first few items to capture variations
            for item in data[:3]:
                recursive_keys(item, f"{path}[]", keys_found)
    
    return keys_found

def inspect_stats_types(data):
    """Specific inspector for the 'statistics' list to see what types are available."""
    stats = data.get('statistics', [])
    types = set()
    for s in stats:
        # Check for nested type object
        t = s.get('type')
        if isinstance(t, dict):
            types.add(t.get('name', 'Unnamed'))
        elif isinstance(t, str):
            types.add(t)
        else:
            # Fallback to check raw keys or ID
            if 'type_id' in s:
                types.add(f"ID:{s['type_id']}")
                
    if not types and stats:
        print(f"   ⚠️ Stats exist but type parsing failed. Sample: {stats[0]}")
        
    return types


def deep_audit(path):
    print(f"🕵️♂️ Deep Audit of {path}")
    
    try:
        with gzip.open(path, 'rt', encoding='UTF-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    # 1. Recursive Key Map
    print("\n1. Data Structure (Unique Paths):")
    # limit depth or noise?
    # keys = recursive_keys(data)
    # for k in sorted(keys.keys()):
    #     print(f"   {k}")
    # Actually, listing ALL keys might be too much. Let's focus on high-value nodes.
    
    high_value_nodes = ['statistics', 'lineups', 'events', 'sidelined', 'referees', 'weather', 'trends']
    for node in high_value_nodes:
        val = data.get(node)
        if val:
            print(f"   ✅ {node}: Found ({len(val) if isinstance(val, list) else 'Dict'})")
        else:
            print(f"   ❌ {node}: Not Found")
            
    # 2. Statistics Breakdown (The Gold Mine)
    print("\n2. Available Match Statistics:")
    stats_types = inspect_stats_types(data)
    if stats_types:
        for t in sorted(list(stats_types)):
            print(f"   - {t}")
    else:
        print("   (No statistics records found)")

    # 3. Lineup Details Breakdown
    print("\n3. Lineup Player Stats Types:")
    lineups = data.get('lineups', [])
    player_stat_types = set()
    if lineups:
        for p in lineups:
            details = p.get('details', [])
            for d in details:
                t = d.get('type', {}).get('name')
                if t:
                    player_stat_types.add(t)
    
    for t in sorted(list(player_stat_types)):
        print(f"   - {t}")

    # 4. Events Breakdown
    print("\n4. Event Types:")
    events = data.get('events', [])
    event_types = set()
    for e in events:
        t = e.get('type', {}).get('name')
        if t:
            event_types.add(t)
            
    for t in sorted(list(event_types)):
        print(f"   - {t}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        deep_audit(sys.argv[1])
    else:
        # Find a recent file
        import glob
        files = glob.glob("data/raw/fixtures/*/*.json.gz")
        if files:
            # Sort by modification time to get newest harvested
            files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            deep_audit(files[0])
        else:
            print("No files found to audit.")
