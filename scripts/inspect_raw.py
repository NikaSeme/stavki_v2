
import sys
import json
import gzip
from pathlib import Path

def inspect(path):
    print(f"🕵️♂️ Inspecting {path}")
    with gzip.open(path, 'rt', encoding='UTF-8') as f:
        data = json.load(f)
    
    # 1. Basic Info
    print(f"ID: {data.get('id')}")
    print(f"Name: {data.get('name')}")
    
    # 4. Participants (Score)
    participants = data.get('participants', [])
    print(f"👥 Participants: {len(participants)}")
    for p in participants:
        print(f"   - Keys: {list(p.keys())}")
        print(f"   - Meta: {p.get('meta')}")
    
    # 5. Scores (Top Level?)
    scores = data.get('scores', [])
    print(f"⚽ Scores: {scores}")
    
    # 6. Result Info
    print(f"🏆 Reference: {data.get('result_info')}")
        
    # 3. Lineups
    lineups = data.get('lineups', [])
    print(f"👕 Lineups: {len(lineups)}")
    if lineups:
        p1 = lineups[0]
        print(f"   Player 1: {p1.get('player', {}).get('display_name')}")
        details = p1.get('details', [])
        print(f"   Details Count: {len(details)}")
        print(f"   Details Sample: {details}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect(sys.argv[1])
    else:
        print("Usage: python3 inspect_raw.py <path_to_json.gz>")
