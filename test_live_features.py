import os
import sys
import logging
from datetime import datetime, timedelta

# Ensure we can import stavki modules
sys.path.append(os.getcwd())

from stavki.data.collectors.sportmonks import SportMonksCollector
from stavki.data.schemas import League
from stavki.pipelines.daily import DailyPipeline
from stavki.features.builders.sm_odds import SMOddsFeatureBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestLiveFeatures")

def main():
    logger.info("Starting Live Feature Verification on VM...")
    
    # 1. Initialize Collector
    try:
        collector = SportMonksCollector()
        logger.info("Collector initialized.")
    except Exception as e:
        logger.error(f"Failed to init collector: {e}")
        return

    # 2. Fetch upcoming matches for a major league (higher chance of markets)
    # EPL or La Liga
    leagues_to_check = [League.EPL, League.LA_LIGA]
    
    matches_found = []
    
    for league in leagues_to_check:
        logger.info(f"Checking {league.value} for upcoming matches...")
        matches = collector.fetch_matches(league, max_hours_ahead=72)
        if matches:
            logger.info(f"Found {len(matches)} matches in {league.value}")
            matches_found.extend(matches[:3]) # Take top 3
        
    if not matches_found:
        logger.warning("No upcoming matches found in major leagues. Trying all...")
        # Fallback logic if needed, but usually there's something.
        
    # 3. Check for Multi-Market Odds
    logger.info(f"Checking {len(matches_found)} matches for multi-market odds...")
    
    corners_count = 0
    btts_count = 0
    
    for m in matches_found:
        logger.info(f"  Match: {m.home_team.name} vs {m.away_team.name} ({m.commence_time})")
        
        try:
            # Manually trigger the fetch logic used in DailyPipeline
            # accessing the internal client directly for granular inspection
            odds = collector.client.get_fixture_odds(int(m.source_id), market="ALL")
            
            has_corners = False
            has_btts = False
            
            for o in odds:
                m_type = o.get("market_type")
                if m_type == "corners_1x2":
                    has_corners = True
                elif m_type == "btts":
                    has_btts = True
            
            if has_corners:
                corners_count += 1
                logger.info("    ✅ FOUND Corners 1X2")
            else:
                logger.info("    ❌ No Corners 1X2")
                
            if has_btts:
                btts_count += 1
                logger.info("    ✅ FOUND BTTS")
            else:
                logger.info("    ❌ No BTTS")
                
        except Exception as e:
            logger.error(f"    Error fetching odds: {e}")

    # 4. Final Report
    logger.info("-" * 40)
    logger.info(f"Checked {len(matches_found)} matches.")
    logger.info(f"Matches with Corners Odds: {corners_count}")
    logger.info(f"Matches with BTTS Odds:    {btts_count}")
    
    if corners_count > 0 or btts_count > 0:
        print("SUCCESS: Live multi-market data is accessible!")
    else:
        print("WARNING: No multi-market data found (could be no markets available yet).")

if __name__ == "__main__":
    main()
