import redis
import sys
import logging

logging.basicConfig(level=logging.WARNING)

from stavki.data.collectors.odds_api import OddsAPIClient
from stavki.data.processors.normalize import normalize_team_name
from stavki.data.schemas import League

def main():
    odds_client = OddsAPIClient()
    r = redis.Redis(decode_responses=True)
    
    elo_teams = set(r.hkeys('stavki:elo'))
    xg_teams = set(r.hkeys('stavki:xg'))
    
    missing_elo = 0
    missing_xg = 0
    total_checked = 0

    print("--- Checking Upcoming API Matches vs Internal Database ---")
    for league, sport in OddsAPIClient.SPORT_KEYS.items():
        if league == League.NBA: continue
        
        resp = odds_client.get_odds(sport)
        if not resp.success or not resp.data:
            continue
            
        for match in resp.data:
            home_raw = match['home_team']
            away_raw = match['away_team']
            
            # Use normalize_team_name, the second argument isn't needed here if it's default False
            home = normalize_team_name(home_raw)
            away = normalize_team_name(away_raw)
            
            total_checked += 2
            
            for team, raw in [(home, home_raw), (away, away_raw)]:
                if team not in elo_teams:
                    print(f"❌ MISSING ELO / NAME MAPPING: '{team}' (Raw API Name: '{raw}') in {league.value}")
                    missing_elo += 1
                if team not in xg_teams:
                    print(f"❌ MISSING xG: '{team}' (Raw API Name: '{raw}') in {league.value}")
                    missing_xg += 1

    print("\n--- Summary ---")
    print(f"Total Teams Checked: {total_checked}")
    print(f"Teams Missing ELO: {missing_elo}")
    print(f"Teams Missing xG: {missing_xg}")
    
    if missing_elo == 0 and missing_xg == 0:
        print("✅ ALL UPCOMING TEAMS PERFECTLY MAPPED TO REDIS STATE!")

if __name__ == "__main__":
    main()
