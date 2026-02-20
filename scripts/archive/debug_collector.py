
import logging
import sys
import os
from datetime import datetime, timedelta

# Force valid PYTHONPATH
sys.path.insert(0, os.getcwd())

# Configure logging to see EVERYTHING
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("stavki")
logger.setLevel(logging.DEBUG)

from stavki.data.collectors.sportmonks import SportMonksCollector, SportMonksClient
from stavki.data.schemas import League
from stavki.config import get_config

def debug_collector():
    print("\n--- Debugging SportMonksCollector ---")
    
    # Initialize
    try:
        config = get_config()
        if not config.sportmonks_api_key:
            print("❌ No API key found in config")
            return
            
        print(f"API Key present: {config.sportmonks_api_key[:5]}...")
        collector = SportMonksCollector()
        
        # Target League: Serie A
        league = League.SERIE_A
        league_id = SportMonksClient.LEAGUE_IDS["SERIE_A"]
        print(f"Target: {league} (ID: {league_id})")
        
        # 1. Test get_upcoming_fixtures directly
        print("\n--- 1. Testing Client.get_upcoming_fixtures ---")
        client = collector.client
        days = 3
        
        # Manually reproduce date logic
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
        print(f"Date Range: {start_date} to {end_date}")
        
        fixtures = client.get_upcoming_fixtures(days=days, league_ids=[league_id])
        print(f"Client found {len(fixtures)} fixtures")
        
        for f in fixtures:
            print(f"   - {f.fixture_id}: {f.home_team} vs {f.away_team} at {f.kickoff} (Status: {f.status})")
            
        # 2. Test Collector.fetch_matches (the full pipeline)
        print("\n--- 2. Testing Collector.fetch_matches ---")
        matches = collector.fetch_matches(league, max_hours_ahead=72)
        print(f"Collector returned {len(matches)} matches")
        
        if len(fixtures) > 0 and len(matches) == 0:
            print("❌ MISMATCH: Client found fixtures but Collector filtered them all out!")
            print("Possible reasons: Date cutoff, SourceNormalizer, or Exception during conversion")
            
            # Debug conversion manually
            print("\n--- Debugging Conversion ---")
            from stavki.data.schemas import Match, Team
            from stavki.data.processors.normalize import SourceNormalizer
            
            cutoff = datetime.now() + timedelta(hours=72)
            print(f"Cutoff time: {cutoff}")
            
            for f in fixtures:
                if f.kickoff > cutoff:
                    print(f"   Skipped {f.fixture_id}: Kickoff {f.kickoff} > Cutoff {cutoff}")
                    continue
                
                try:
                    norm_home = SourceNormalizer.normalize(f.home_team)
                    norm_away = SourceNormalizer.normalize(f.away_team)
                    print(f"   Converted {f.fixture_id}: {f.home_team}->{norm_home}, {f.away_team}->{norm_away}")
                except Exception as e:
                    print(f"   ❌ Conversion Status failed for {f.fixture_id}: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_collector()
