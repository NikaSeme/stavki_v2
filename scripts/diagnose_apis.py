
import os
import sys
import logging
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
        import requests
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
                    
                # Now try to fetch odds for EPL
                odds_url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds?regions=uk&markets=h2h&apiKey={key}"
                r_odds = requests.get(odds_url, timeout=10)
                if r_odds.status_code == 200:
                    odds = r_odds.json()
                    print(f"   Fetched {len(odds)} upcoming matches with odds")
                    if odds:
                        m = odds[0]
                        print(f"   Sample: {m.get('home_team')} vs {m.get('away_team')} ({m.get('commence_time')})")
                else:
                    print(f"   ❌ Failed to fetch EPL odds: {r_odds.status_code}")
            else:
                print("   ⚠️ EPL ('soccer_epl') not found in active sports list")
        else:
            print(f"❌ API Error: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

def test_sportmonks():
    """Test SportMonks API connectivity."""
    print("\n--- 3. Testing SportMonks API ---")
    key = os.getenv("SPORTMONKS_API_KEY")
    if not key:
        print("Skipping (no key)")
        return

    try:
        import requests
        # Test basic connectivity (leagues endpoint)
        url = "https://api.sportmonks.com/v3/football/leagues"
        r = requests.get(url, params={"api_token": key}, timeout=10)
        
        if r.status_code == 200:
            data = r.json().get('data', [])
            print(f"✅ Success! Connectivity working (found {len(data)} leagues)")
            
            # Check EPL specifically (ID 8)
            epl_id = 8
            url_fixtures = f"https://api.sportmonks.com/v3/football/fixtures/upcoming/markets/{epl_id}"
            # Actually, standard fixtures endpoint is better
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            url_fixtures = f"https://api.sportmonks.com/v3/football/fixtures/between/{start_date}/{end_date}"
            # Filter by league ID is tricky in V3 without includes, trying simpler endpoint
            
            # Using client wrapper logic to reproduce collector behavior
            params = {
                "api_token": key,
                "include": "league;participants;venue",
                "leagues": epl_id  # Filter if supported
            }
             
            print(f"Checking fixtures between {start_date} and {end_date} for League ID {epl_id}...")
            # Note: For SportMonks V3, filtering by league_id often happens via include or specific endpoint
            # We'll fetch date range and filter client-side
            matches_url = f"https://api.sportmonks.com/v3/football/fixtures/between/{start_date}/{end_date}"
            matches_params = {
                "api_token": key,
                "include": "participants;venue;league"
            }
            r_fixt = requests.get(matches_url, params=matches_params, timeout=10)
            
            if r_fixt.status_code == 200:
                fixtures = r_fixt.json().get('data', [])
                # Client-side filter for league ID
                epl_fixtures = [f for f in fixtures if f.get('league_id') == epl_id]
                print(f"   Found {len(fixtures)} total fixtures, {len(epl_fixtures)} for EPL")
                
                if epl_fixtures:
                    f = epl_fixtures[0]
                    home = next((p['name'] for p in f.get('participants', []) if p['meta']['location'] == 'home'), 'Home')
                    away = next((p['name'] for p in f.get('participants', []) if p['meta']['location'] == 'away'), 'Away')
                    print(f"   Sample: {home} vs {away} at {f.get('starting_at')}")
            else:
                print(f"   ❌ Failed to fetch fixtures: {r_fixt.status_code} - {r_fixt.text[:100]}")

        else:
             print(f"❌ API Error: {r.status_code} - {r.text[:200]}...")
             
    except Exception as e:
        print(f"❌ SportMonks check failed: {e}")

if __name__ == "__main__":
    if check_env():
        test_odds_api()
        test_sportmonks()
    print("\nDone.")
