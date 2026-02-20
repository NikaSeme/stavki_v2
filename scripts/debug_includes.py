
import sys
from pathlib import Path
import logging

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from stavki.data.collectors.sportmonks import SportMonksClient
from stavki.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
# USER KEY
API_KEY = "wFzsoRjY2uZfEpMhUDPxwLBw9o7JJVIZi1AjolrzN7Wqda0OjeTh32XAgZOB"
TARGET_ID = 18850186 # The one that worked before (Betis)

def debug_includes():
    client = SportMonksClient(api_key=API_KEY)
    
    print(f"🕵️♂️ Debugging Includes for Fixture {TARGET_ID}")
    
    candidates = [
        "scores"
    ]
    
    for inc in candidates:
        print(f"\nTesting include: '{inc}' ...")
        resp = client._request(f"fixtures/{TARGET_ID}", includes=[inc])
        
        if resp.get("error"):
            print(f"   ❌ FAILED (404/Error): {resp['error']}")
        else:
            # Check if data actually came back
            data = resp.get("data", {})
            # extracting the key might be tricky, usually it's the first part of include
            key = inc.split('.')[0]
            val = data.get(key)
            if val:
                print(f"   ✅ SUCCESS: Found {len(val) if isinstance(val, list) else 'Dict'}")
            else:
                print(f"   ⚠️ SUCCESS (No Error) but key '{key}' is empty/missing in response.")

if __name__ == "__main__":
    debug_includes()
