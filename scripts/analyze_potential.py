
import sys
from pathlib import Path
import logging
import json
import requests

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stavki.data.collectors.sportmonks import SportMonksClient

logging.basicConfig(level=logging.INFO)

# USER PROVIDED KEY
USER_KEY = "wFzsoRjY2uZfEpMhUDPxwLBw9o7JJVIZi1AjolrzN7Wqda0OjeTh32XAgZOB"
TARGET_FIXTURE_ID = 19439492

def probe_api():
    print("🕵️♂️ BOB'S GOLD MINE PROBE")
    print("="*60)
    print(f"Target Fixture: {TARGET_FIXTURE_ID}")
    
    client = SportMonksClient(api_key=USER_KEY)
    
    # 1. Test User's Specific URL Includes
    print("\n1. Testing User's URL Includes (Sidelined)...")
    includes = [
        "sidelined.sideline.player",
        "sidelined.sideline.type",
        "participants",
        "league",
        "venue",
        "state"
    ]
    
    data = client._request(f"fixtures/{TARGET_FIXTURE_ID}", includes=includes)
    
    if data.get("error"):
        print(f"   ❌ Failed: {data['error']}")
    else:
        item = data.get("data", {})
        home = next((p['name'] for p in item.get('participants', []) if p['meta']['location'] == 'home'), 'Unknown')
        away = next((p['name'] for p in item.get('participants', []) if p['meta']['location'] == 'away'), 'Unknown')
        print(f"   ✅ Match: {home} vs {away}")
        
        # Check Sidelined (Injuries)
        sidelined = item.get("sidelined", [])
        print(f"   🚑 Sidelined/Injured: {len(sidelined)}")
        for s in sidelined:
            player = s.get("sideline", {}).get("player", {}).get("display_name", "Unknown")
            reason = s.get("sideline", {}).get("type", {}).get("name", "Unknown")
            print(f"      - {player}: {reason}")

    # 2. Test Statistics Wealth
    print("\n2. Testing Statistics Wealth...")
    data = client._request(f"fixtures/{TARGET_FIXTURE_ID}", includes=["statistics"])
    if data.get("data"):
        stats = data.get("data", {}).get("statistics", [])
        print(f"   📊 Statistics Points: {len(stats)}")
        # Sample types
        types = set()
        for s in stats:
             t = s.get("type", {})
             if isinstance(t, dict):
                 types.add(t.get("name"))
        print(f"   Sample Types: {list(types)[:5]}...")
    
    # 3. Test Lineup Details (Ratings)
    print("\n3. Testing Lineup Details (Ratings)...")
    data = client._request(f"fixtures/{TARGET_FIXTURE_ID}", includes=["lineups.details.type", "lineups.player"])
    if data.get("data"):
         lineups = data.get("data", {}).get("lineups", [])
         print(f"   👕 Linup Entries: {len(lineups)}")
         has_details = any(l.get("details") for l in lineups)
         if has_details:
             print("   ✅ Player Details (Ratings/Shots) FOUND!")
         else:
             print("   ⚠️ No player details found.")

if __name__ == "__main__":
    probe_api()
