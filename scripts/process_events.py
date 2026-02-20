
import sys
import json
import gzip
import logging
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_events():
    raw_dir = PROJECT_ROOT / "data" / "raw" / "fixtures"
    files = list(raw_dir.rglob("*.json.gz"))
    
    logger.info(f"Found {len(files)} raw fixture files.")
    
    full_events = []
    
    # Regex Patterns
    # Note: comments vary by league provider, but common patterns exist
    # "Player swap - Team. Out is replaced by In due to an injury."
    re_injury = re.compile(r"due to an injury", re.IGNORECASE)
    
    # "Penalty conceded by Player (Team)."
    # "Penalty saved!"
    # "Goal! Team 1, Team 0. Player (Team) converts the penalty..."
    re_penalty_miss = re.compile(r"penalty.*(saved|missed)", re.IGNORECASE)
    re_penalty_goal = re.compile(r"converts the penalty", re.IGNORECASE)
    
    # "Red Card shown to Player (Team)."
    # "Second yellow card to Player (Team)."
    re_red_card = re.compile(r"(red card|second yellow)", re.IGNORECASE)
    
    # "VAR Decision: ..."
    re_var = re.compile(r"VAR Decision", re.IGNORECASE)
    
    # "Own Goal by ..."
    re_own_goal = re.compile(r"own goal", re.IGNORECASE)
    
    for fpath in tqdm(files, desc="Mining Events"):
        try:
            with gzip.open(fpath, 'rt', encoding='UTF-8') as f:
                data = json.load(f)
                
            fid = data.get('id')
            start_at = data.get('starting_at')
            if not start_at: continue
            date_val = pd.to_datetime(start_at).date()
            
            # Identify teams
            participants = data.get('participants', [])
            home = next((p for p in participants if p['meta']['location'] == 'home'), {})
            away = next((p for p in participants if p['meta']['location'] == 'away'), {})
            
            home_id = home.get('id')
            away_id = away.get('id')
            home_name = home.get('name', 'Home')
            away_name = away.get('name', 'Away')
            
            comments = data.get('comments', [])
            
            # Match Level Flags
            events = {
                'match_id': fid,
                'date': date_val,
                'home_team_id': home_id,
                'away_team_id': away_id,
                'home_red_cards': 0,
                'away_red_cards': 0,
                'home_injuries': 0,
                'away_injuries': 0,
                'home_penalties': 0,
                'away_penalties': 0,
                'home_own_goals': 0,
                'away_own_goals': 0,
                'var_interventions': 0
            }
            
            for c in comments:
                txt = c.get('comment', '')
                if not txt: continue
                txt_lower = txt.lower()
                
                # Assign to team?
                # Sometimes team name is in text: "Player swap - Blackburn Rovers."
                is_home = home_name.lower() in txt_lower
                is_away = away_name.lower() in txt_lower
                
                # If both or neither, hard to say. Skip or look for other cues?
                # Often comment has "Player (Team)" format.
                
                # 1. Injuries
                if re_injury.search(txt):
                    if is_home: events['home_injuries'] += 1
                    elif is_away: events['away_injuries'] += 1
                    
                # 2. Red Cards
                if re_red_card.search(txt):
                    if is_home: events['home_red_cards'] += 1
                    elif is_away: events['away_red_cards'] += 1
                    
                # 3. Penalties
                # Conceded or Won? Usually 'converts' means scored.
                if re_penalty_goal.search(txt):
                    if is_home: events['home_penalties'] += 1
                    elif is_away: events['away_penalties'] += 1
                elif re_penalty_miss.search(txt):
                    # Missed/Saved penalty = penalty attempt
                    if is_home: events['home_penalties'] += 1
                    elif is_away: events['away_penalties'] += 1
                    
                # 4. Own Goals
                # "Own Goal by Player (Team)" -> Affects OTHER team scores
                if re_own_goal.search(txt):
                    # If home team scored own goal -> Away benefit, but Home *committed* it
                    # We want "clumsiness" context.
                    if is_home: events['home_own_goals'] += 1
                    elif is_away: events['away_own_goals'] += 1
                    
                # 5. VAR
                if re_var.search(txt):
                    events['var_interventions'] += 1
            
            full_events.append(events)
                
        except Exception as e:
            logger.error(f"Error processing {fpath}: {e}")
            
    if full_events:
        df = pd.DataFrame(full_events)
        out_path = PROJECT_ROOT / "data" / "processed" / "matches" / "match_events_silver.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path)
        logger.info(f"Saved {len(df)} event records to {out_path}")
        print(df.head())
    else:
        logger.warning("No events found.")

if __name__ == "__main__":
    process_events()
