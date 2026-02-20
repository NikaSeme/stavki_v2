import sys
import os
import json
import logging

# Add project root
sys.path.insert(0, os.getcwd())

from stavki.config import get_config
from stavki.data.collectors.sportmonks import SportMonksClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_odds():
    config = get_config()
    client = SportMonksClient(api_key=config.sportmonks_api_key)
    
    fixture_id = 19425130 # Udinese vs Sassuolo
    print(f"Fetching odds for fixture {fixture_id}...")
    
    # Raw request to see exactly what comes back
    response = client._request(
        f"fixtures/{fixture_id}",
        includes=["odds.bookmaker", "odds.market"] 
    )
    
    data = response.get("data", {})
    odds = data.get("odds", [])
    
    print(f"Total Odds Records: {len(odds)}")
    
    if not odds:
        print("No odds found in response!")
        return

    # Scan for Fulltime Result specifically
    print("\n--- Inspecting 'Fulltime Result' Group (Bookmaker 57 - Cashpoint) ---")
    
    cashpoint_odds = []
    for o in odds:
        m_name = o.get("market", {}).get("name", "")
        b_id = o.get("bookmaker_id")
        
        if m_name == "Fulltime Result" and b_id == 57:
            cashpoint_odds.append(o)
            
    print(f"Found {len(cashpoint_odds)} odds for Cashpoint Fulltime Result:")
    for o in cashpoint_odds:
        print(f"Label: '{o.get('label')}' | Value: {o.get('value')} | Name: '{o.get('name')}'")

    # Test built-in method
    print("\n--- Testing client.get_fixture_odds ---")
    processed = client.get_fixture_odds(fixture_id)
    print(f"Processed results: {len(processed)} bookmakers")

if __name__ == "__main__":
    debug_odds()
