
import os
import sys
import logging
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Setup minimal logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnostic")

def check_env():
    """Check environment variables."""
    print("\n--- 1. Checking Environment ---")
    load_dotenv()
    
    odds_key = os.getenv("ODDS_API_KEY")
    sm_key = os.getenv("SPORTMONKS_API_KEY")
    
    print(f"ODDS_API_KEY:       {'[SET]' if odds_key else '[MISSING]'}")
    print(f"SPORTMONKS_API_KEY: {'[SET]' if sm_key else '[MISSING]'}")
    
    if not odds_key and not sm_key:
        print("❌ No API keys found! Check your .env file.")
        return False
    return True

def test_odds_api():
    """Test The Odds API connectivity."""
    print("\n--- 2. Testing The Odds API ---")
    key = os.getenv("ODDS_API_KEY")
    if not key:
        print("Skipping (no key)")
        return

    try:
        # Test basic sports list (lightweight)
        url = f"https://api.the-odds-api.com/v4/sports?apiKey={key}"
        r = requests.get(url, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            print(f"✅ Success! Found {len(data)} active sports")
            
            # Check specifically for EPL
            epl = next((s for s in data if s['key'] == 'soccer_epl'), None)
            if epl:
                print(f"   Found EPL: {epl['title']} ({epl['key']})")
                if not epl.get('active', True):
                    print("   ⚠️ EPL is marked as INACTIVE (no odds usually available)")
                else:
                    print("   ✅ EPL is ACTIVE")
            else:
                print("   ⚠️ EPL ('soccer_epl') not found in active sports list")
        else:
            print(f"❌ API Error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

def get_league_name(key, league_id):
    """Helper to fetch league name by ID."""
    try:
        url = f"https://api.sportmonks.com/v3/football/leagues/{league_id}"
        r = requests.get(url, params={"api_token": key}, timeout=5)
        if r.status_code == 200:
            return r.json().get('data', {}).get('name', f"Unknown-{league_id}")
    except:
        pass
    return f"ID-{league_id}"

def test_sportmonks():
    """Test SportMonks API connectivity and ID mismatch."""
    print("\n--- 3. Testing SportMonks API ---")
    key = os.getenv("SPORTMONKS_API_KEY")
    if not key:
        print("Skipping (no key)")
        return

    try:
        # 1. Search for "Premier League" explicitly to find correct ID
        print("Searching for 'Premier League' to confirm League ID...")
        url_leagues = "https://api.sportmonks.com/v3/football/leagues"
        r_l = requests.get(url_leagues, params={"api_token": key}, timeout=10)
        
        real_epl_id = None
        if r_l.status_code == 200:
            leagues = r_l.json().get('data', [])
            # Search for England Premier League
            for l in leagues:
                # Need to be careful with matching, many "Premier Leagues"
                # V3 usually includes country info if requested, but here we scan names
                if "Premier League" in l.get("name", ""):
                    # In V3, checking country requires another call or include, 
                    # but let's just print candidates
                    print(f"   Candidate: {l.get('name')} (ID: {l.get('id')})")
                    if l.get('name') == "Premier League": # Exact match often implies England
                        real_epl_id = l.get('id')
        
        # Default V2/V3 ID for EPL is often 8, but let's see
        config_id = 8
        target_id = real_epl_id if real_epl_id else config_id
        
        print(f"Using Target League ID: {target_id} (Config default: {config_id})")
        if real_epl_id and real_epl_id != config_id:
             print(f"⚠️ MISMATCH DETECTED! Config uses {config_id}, API says {real_epl_id}")

        # 2. Check fixtures in date range
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
        
        print(f"Fetching fixtures between {start_date} and {end_date}...")
        
        matches_url = f"https://api.sportmonks.com/v3/football/fixtures/between/{start_date}/{end_date}"
        # We assume V3. Using include=league to see names
        matches_params = {
            "api_token": key,
            "include": "league"
        }
        
        r_fixt = requests.get(matches_url, params=matches_params, timeout=10)
        
        if r_fixt.status_code == 200:
            fixtures = r_fixt.json().get('data', [])
            print(f"✅ Found {len(fixtures)} total fixtures in next 7 days")
            
            # Count by league
            league_counts = {}
            for f in fixtures:
                lid = f.get('league_id')
                if lid not in league_counts:
                    league_name = "Unknown"
                    if 'league' in f and f['league']:
                        league_name = f['league'].get('name', 'Unknown')
                    league_counts[lid] = {'name': league_name, 'count': 0}
                league_counts[lid]['count'] += 1
            
            # Print summary
            print("\nFixtures found by League:")
            for lid, info in league_counts.items():
                print(f"   - ID {lid}: {info['name']} ({info['count']} matches)")
                
            # Check if our target ID is there
            if target_id in league_counts:
                print(f"\n✅ SUCCESS: Found {league_counts[target_id]['count']} fixtures for League ID {target_id}")
            else:
                print(f"\n❌ ERROR: No fixtures found for Target League ID {target_id}")
                
        else:
             print(f"❌ API Error: {r_fixt.status_code} - {r_fixt.text[:200]}...")
             
    except Exception as e:
        print(f"❌ SportMonks check failed: {e}")

if __name__ == "__main__":
    if check_env():
        test_odds_api()
        test_sportmonks()
    print("\nDone.")
