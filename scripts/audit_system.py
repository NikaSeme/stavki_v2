
import sys
import os
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Force valid PYTHONPATH
sys.path.insert(0, os.getcwd())

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit")

def audit_system():
    print("\n🔍 STAVKI System Audit")
    print("=======================")
    
    # 1. Environment & Dependencies
    print("\n--- 1. Environment Check ---")
    load_dotenv()
    
    keys = {
        "ODDS_API_KEY": os.getenv("ODDS_API_KEY"),
        "SPORTMONKS_API_KEY": os.getenv("SPORTMONKS_API_KEY"),
    }
    
    for k, v in keys.items():
        status = "✅ Set" if v else "❌ MISSING"
        print(f"{k}: {status}")
    
    try:
        import pydantic
        print(f"Pydantic: ✅ Installed ({pydantic.VERSION})")
    except ImportError:
        print("Pydantic: ❌ MISSING")
        
    try:
        import stavki
        print(f"Stavki Package: ✅ Importable ({stavki.__file__})")
    except ImportError:
        print("Stavki Package: ❌ IMPORT FAILED")
        return

    # 2. Collector Logic Simulation
    print("\n--- 2. Pipeline Simulation (DailyPipeline) ---")
    
    # Configure logging to see our debug messages
    logging.basicConfig(level=logging.DEBUG)
    # Filter out noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    try:
        from stavki.data.collectors.sportmonks import SportMonksCollector, SportMonksClient
        from stavki.data.schemas import League
        
        # Initialize
        print("Initializing SportMonksCollector...")
        sm_collector = SportMonksCollector()
        print("✅ Collector initialized")
        
        # Test Fetch Matches
        league_enum = League.SERIE_A
        print(f"\nStep A: Fetching matches for {league_enum}...")
        matches = sm_collector.fetch_matches(league_enum, max_hours_ahead=72)
        
        print(f"→ Found {len(matches)} matches")
        if not matches:
             print("❌ Failure: No matches found.")
        else:
            for i, m in enumerate(matches[:3]):
                 print(f"   {i+1}. {m.home_team.name} vs {m.away_team.name} (ID: {m.id}, SourceID: {m.source_id})")

            # Test Fetch Odds
            print(f"\nStep B: Fetching odds for {len(matches)} matches...")
            odds_results = sm_collector.fetch_odds(league_enum, matches=matches)
            
            print(f"→ Returned odds for {len(odds_results)} matches")
            
            # Inspect Odds
            valid_odds_count = 0
            for match_id, snapshots in odds_results.items():
                if snapshots:
                    valid_odds_count += 1
                    snap = snapshots[0]
                    print(f"   Match {match_id}: {snap.bookmaker} -> H:{snap.home_odds} D:{snap.draw_odds} A:{snap.away_odds}")
                else:
                    print(f"   Match {match_id}: No snapshots returned (empty list)")
                    
            if valid_odds_count == 0:
                print("❌ Failure: No valid odds found for any match.")
                print("   Possible reasons: Market data missing in API, or parsing error.")
                
                # Deep debug odds for first match
                if matches:
                     m = matches[0]
                     print(f"\n   Deep Debug: Raw odds response for Match {m.source_id}...")
                     
                     try:
                         # Force logging to DEBUG again after imports might have reset it
                         logging.getLogger().setLevel(logging.DEBUG)
                         logging.getLogger("stavki").setLevel(logging.DEBUG)
                         
                         # Manually request with same include as collector
                         raw_resp = sm_collector.client._request(
                            f"fixtures/{m.source_id}",
                            includes=["odds.bookmaker", "odds.market"] 
                         )
                         print(f"   API Response Keys: {list(raw_resp.keys())}")
                         
                         data = raw_resp.get("data", {})
                         all_odds = data.get("odds", [])
                         print(f"   Total Odds Records: {len(all_odds)}")
                         
                         if all_odds:
                             print(f"   Scanning all {len(all_odds)} records for 1X2/Fulltime Result...")
                             
                             coverage = {} # { bookmaker: { home: val, draw: val, away: val } }
                             
                             # Iterate over ALL records to ensure we find matching parts
                             for o in all_odds:
                                 mkt_name = o.get("market", {}).get("name", "Unknown")
                                 bk_name = o.get("bookmaker", {}).get("name", "Unknown")
                                 
                                 is_target = False
                                 if "1X2" in mkt_name or "Fulltime Result" in mkt_name or "Match Winner" in mkt_name or "3Way Result" in mkt_name:
                                     is_target = True
                                 
                                 if is_target:
                                     if bk_name not in coverage:
                                         coverage[bk_name] = {}
                                         
                                     label = o.get("label", "").lower()
                                     val = o.get("value")
                                     
                                     if label in ["1", "home"]:
                                         coverage[bk_name]["home"] = val
                                     elif label in ["x", "draw"]:
                                         coverage[bk_name]["draw"] = val
                                     elif label in ["2", "away"]:
                                         coverage[bk_name]["away"] = val
                             
                             print("\n   --- Bookmaker Coverage Analysis ---")
                             complete_count = 0
                             for bk, outcomes in coverage.items():
                                 has_h = "home" in outcomes
                                 has_d = "draw" in outcomes
                                 has_a = "away" in outcomes
                                 
                                 status = "✅ COMPLETE" if (has_h and has_d and has_a) else "⚠️ PARTIAL"
                                 if status == "✅ COMPLETE":
                                     complete_count += 1
                                     
                                 print(f"   {bk}: {status} [H:{outcomes.get('home')} D:{outcomes.get('draw')} A:{outcomes.get('away')}]")
                                 
                             if complete_count == 0:
                                 print("\n   ❌ CRITICAL: No bookmaker has complete odds set!")
                             else:
                                 print(f"\n   ✅ FOUND {complete_count} bookmakers with complete odds sets.")
                         else:
                             print("   ❌ No odds data found in 'odds' field. Check plan/includes.")
                             
                     except Exception as e:
                         print(f"   Error fetching raw odds: {e}")

            # 3. Pipeline Merge Logic (simulate daily.py)
            print("\n--- 3. Pipeline Merge Simulation ---")
            rows = []
            for m in matches:
                best_snap = None
                if m.id in odds_results and odds_results[m.id]:
                    best_snap = odds_results[m.id][0]
                
                status = "✅ Included" if best_snap else "❌ Dropped (No Odds)"
                print(f"Match {m.id}: {status}")
                
                if best_snap:
                    rows.append({"event_id": m.id})
                    
            if not rows:
                print("\n❌ Pipeline Result: 0 matches with odds (DailyPipeline would return empty DF)")
            else:
                print(f"\n✅ Pipeline Result: {len(rows)} matches ready for prediction")

    except Exception as e:
        print(f"\n❌ CRITICAL EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    audit_system()
