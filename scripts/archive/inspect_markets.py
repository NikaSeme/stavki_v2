#!/usr/bin/env python3
"""
Inspect raw market names for a specific team (e.g. Toulouse) to see why non-1X2 markets are being captured.
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stavki.config import get_config
from stavki.data.collectors import SportMonksCollector
from stavki.data.schemas import League

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inspect_markets")

def main():
    target_team = "Toulouse"
    if len(sys.argv) > 1:
        target_team = sys.argv[1]
        
    logger.info(f"Searching for match involving: {target_team}")
    
    # Init collector
    config = get_config()
    sm_collector = SportMonksCollector()
    
    # Needs to search efficiently... currently collectors require a League.
    # We'll just search Ligue 1 (Toulouse is in Ligue 1)
    # If not found, user might need to specify league, but I'll assume Ligue 1 for now based on "Toulouse".
    
    league = League.LIGUE_1
    matches = sm_collector.fetch_matches(league, max_hours_ahead=168) # 7 days
    
    target_match = None
    for m in matches:
        if target_team.lower() in m.home_team.name.lower() or target_team.lower() in m.away_team.name.lower():
            target_match = m
            break
            
    if not target_match:
        logger.error(f"Match not found for {target_team} in Ligue 1")
        return

    logger.info(f"Found match: {target_match.home_team.name} vs {target_match.away_team.name} (ID: {target_match.source_id})")
    
    # Fetch ALL odds raw
    logger.info("Fetching raw odds data...")
    raw_odds = sm_collector.client.get_fixture_odds(int(target_match.source_id), market="ALL")
    
    print("\n" + "="*80)
    print(f"RAW MARKET INSPECTION")
    print("="*80)
    
    # Replicate the logic from sportmonks.py to see what PASSES
    valid_aliases = ["1x2", "fulltime result", "match winner", "3way result", "match result"]
    exclude_terms = ["corner", "card", "half", "period", "handicap", "booking", "goal"]

    seen_markets = set()
    
    for o in raw_odds:
        market_name = o.get("market", {}).get("name", "")
        if not market_name: continue
        
        normalized_name = market_name.lower()
        
        # Check if it passes current filter
        passed = False
        if any(alias in normalized_name for alias in valid_aliases):
            if not any(ex in normalized_name for ex in exclude_terms):
                passed = True
        
        # Display
        if market_name not in seen_markets:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} | {market_name}")
            
            # If it passed but shouldn't have (contains suspected noise)
            if passed:
                # Check for suspicious terms
                suspicious = ["score", "btts", "combo", "double", "winner", "margin"]
                if any(s in normalized_name for s in suspicious) and "match winner" not in normalized_name:
                     print(f"      ^--- SUSPICIOUS! (might be causing the 8.8 odds)")
                     
                # Print sample values to see if they look like 8.8
                values = [o.get("value") for o in raw_odds if o.get("market", {}).get("name") == market_name][:3]
                print(f"      Sample values: {values}")
            
            seen_markets.add(market_name)

if __name__ == "__main__":
    main()
