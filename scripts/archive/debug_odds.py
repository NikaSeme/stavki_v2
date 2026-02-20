#!/usr/bin/env python3
"""
Debug script to fetch LIVE odds for a specific league/match and print them.
Usage: python scripts/debug_odds.py [league_name]
Example: python scripts/debug_odds.py soccer_epl
"""

import sys
import os
import logging
import asyncio
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stavki.config import get_config
from stavki.data.collectors import SportMonksCollector, OddsAPICollector
from stavki.data.schemas import League

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_odds")

def main():
    league_str = sys.argv[1] if len(sys.argv) > 1 else "soccer_epl"
    
    # Map string to League enum
    league_map = {
        "soccer_epl": League.EPL,
        "epl": League.EPL,
        "soccer_spain_la_liga": League.LA_LIGA,
        "laliga": League.LA_LIGA,
        "soccer_germany_bundesliga": League.BUNDESLIGA,
        "bundesliga": League.BUNDESLIGA,
        "soccer_italy_serie_a": League.SERIE_A,
        "seriea": League.SERIE_A,
    }
    
    league = league_map.get(league_str.lower())
    if not league:
        logger.error(f"Unknown league: {league_str}")
        return

    logger.info(f"Fetching live data for {league.name}...")
    
    # Init collectors
    config = get_config()
    sm_collector = SportMonksCollector() if config.sportmonks_api_key else None
    
    if not sm_collector:
        logger.error("SportMonks API key not found.")
        return

    # 1. Fetch Fixtures
    logger.info("1. Fetching fixtures...")
    matches = sm_collector.fetch_matches(league, max_hours_ahead=48)
    logger.info(f"Found {len(matches)} upcoming matches.")
    
    if not matches:
        return

    # 2. Fetch Odds
    logger.info("2. Fetching live odds (ALL markets)...")
    # Fetch for first 3 matches only to save requests/time
    target_matches = matches[:3]
    
    odds_data = sm_collector.fetch_odds(league, matches=target_matches)
    
    # 3. Display
    print("\n" + "="*80)
    print(f"LIVE ODDS REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    for match in target_matches:
        print(f"Match: {match.home_team.name} vs {match.away_team.name}")
        print(f"Kickoff: {match.commence_time}")
        print("-" * 40)
        
        snapshots = odds_data.get(match.id, [])
        if not snapshots:
            print("  No odds found.")
            continue
            
        # Group by Bookie
        print(f"  {'Bookmaker':<20} | {'Home':<6} | {'Draw':<6} | {'Away':<6} | {'Updated At':<20}")
        print("  " + "-"*70)
        
        for snap in snapshots:
            # Show ALL markets now
            print(f"  {snap.bookmaker:<20} | {snap.market:<15} | {snap.home_odds:<6.2f} | {str(snap.draw_odds):<6} | {snap.away_odds:<6.2f} | {snap.timestamp.strftime('%H:%M:%S')}")
        
        print("\n")

if __name__ == "__main__":
    main()
