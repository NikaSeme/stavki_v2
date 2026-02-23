#!/usr/bin/env python3
"""
Cache Live State (Redis Serializer)
===================================
Executes at the end of the daily pipeline to compute the final, rolling 
state of all 23,000 matches and serializes the dictionaries into a fast
O(1) Redis In-Memory Cache.

This drops live inference lag for the Telegram Bot near 0ms
by bypassing the 5-second `features_full.csv` loading step entirely.
"""

import sys
import json
import logging
from pathlib import Path
import pandas as pd
import redis

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stavki.config import DATA_DIR
from stavki.data.processors.normalize import normalize_team_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("redis_cache")

def get_redis_client():
    """Establish Redis connection."""
    try:
        client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        # Test connection
        client.ping()
        return client
    except redis.exceptions.ConnectionError:
        logger.error("❌ Redis server is not running! Please run `brew services start redis`")
        sys.exit(1)

def cache_live_state():
    r = get_redis_client()
    logger.info("✅ Connected to Redis")

    # Clear old keys to prevent ghost records
    keys = r.keys('stavki:*')
    if keys:
        r.delete(*keys)
        logger.info(f"🧹 Cleared {len(keys)} old Redis namespaces")

    # 1. Load the pristine tabular data
    features_csv = DATA_DIR / 'features_full.csv'
    enriched_csv = DATA_DIR / 'features_full_enriched.csv'
    
    src = enriched_csv if enriched_csv.exists() else features_csv
    logger.info(f"Loading matrix from {src.name}...")
    df = pd.read_csv(src, low_memory=False)
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date')
    
    # Normalize keys identically to live inference
    if 'HomeTeam' in df.columns:
        df['HomeTeam'] = df['HomeTeam'].apply(normalize_team_name)
    if 'AwayTeam' in df.columns:
        df['AwayTeam'] = df['AwayTeam'].apply(normalize_team_name)

    logger.info("Serializing to Redis...")

    # ==========================
    # 1. ELO Ratings (HSET)
    # ==========================
    h_elo = df[['HomeTeam', 'elo_home']].rename(columns={'HomeTeam': 'Team', 'elo_home': 'Elo'}).dropna()
    a_elo = df[['AwayTeam', 'elo_away']].rename(columns={'AwayTeam': 'Team', 'elo_away': 'Elo'}).dropna()
    all_elo = pd.concat([h_elo, a_elo]).drop_duplicates(subset=['Team'], keep='last')
    
    elo_dict = all_elo.set_index('Team')['Elo'].to_dict()
    if elo_dict:
        r.hset('stavki:elo', mapping=elo_dict)
    logger.info(f"-> Cached ELO for {len(elo_dict)} teams")

    # ==========================
    # 2. Form History (JSON Strings)
    #    (Last 5 points)
    # ==========================
    team_form = {}
    h_col = "HomeTeam"
    a_col = "AwayTeam"
    subset = df[[h_col, a_col, 'form_home_pts', 'form_away_pts']].tail(3000)
    for row in subset.itertuples(index=False):
        h, a = getattr(row, h_col, ""), getattr(row, a_col, "")
        hp, ap = row.form_home_pts, row.form_away_pts
        if h:
            team_form.setdefault(h, []).append(hp)
        if a:
            team_form.setdefault(a, []).append(ap)

    form_dict = {k: json.dumps(v[-5:]) for k, v in team_form.items()} # Take last 5 and dict
    if form_dict:
        r.hset('stavki:form', mapping=form_dict)
    logger.info(f"-> Cached Form for {len(form_dict)} teams")

    # ==========================
    # 3. Formations History
    # ==========================
    if 'formation_home' in df.columns:
        h_fmt = df[['HomeTeam', 'formation_home']].rename(columns={'HomeTeam': 'Team', 'formation_home': 'Fmt'}).dropna()
        a_fmt = df[['AwayTeam', 'formation_away']].rename(columns={'AwayTeam': 'Team', 'formation_away': 'Fmt'}).dropna()
        all_fmt = pd.concat([h_fmt, a_fmt]).tail(5000)
        fmt_series = all_fmt.groupby('Team')['Fmt'].agg(list)
        fmt_lists = fmt_series.apply(lambda x: json.dumps([str(i) for i in x[-10:]])).to_dict()
        if fmt_lists:
            r.hset('stavki:formations', mapping=fmt_lists)
        logger.info(f"-> Cached Formations for {len(fmt_lists)} teams")

    # ==========================
    # 4. Standard Metrics (xG, Ratings)
    # ==========================
    def cache_metric(col_h, col_a, redis_key):
        if col_h not in df.columns:
            return
        h = df[['HomeTeam', col_h]].rename(columns={'HomeTeam': 'Team', col_h: 'Val'}).dropna()
        a = df[['AwayTeam', col_a]].rename(columns={'AwayTeam': 'Team', col_a: 'Val'}).dropna()
        m = pd.concat([h, a]).drop_duplicates('Team', keep='last')
        mapping = m.set_index('Team')['Val'].to_dict()
        if mapping:
            r.hset(f'stavki:{redis_key}', mapping=mapping)
            logger.info(f"-> Cached {redis_key} for {len(mapping)} teams")

    cache_metric('xg_home', 'xg_away', 'xg')
    cache_metric('avg_rating_home', 'avg_rating_away', 'avg_rating')
    cache_metric('rolling_fouls_home', 'rolling_fouls_away', 'rolling_fouls')
    cache_metric('rolling_yellows_home', 'rolling_yellows_away', 'rolling_yellows')
    cache_metric('rolling_corners_home', 'rolling_corners_away', 'rolling_corners')
    cache_metric('rolling_possession_home', 'rolling_possession_away', 'rolling_possession')

    # ==========================
    # 5. Referee Profiles
    # ==========================
    if 'Referee' in df.columns:
        valid_refs = df[df['Referee'].notna() & (df['Referee'] != '')].copy()
        valid_refs['Referee'] = valid_refs['Referee'].astype(str).str.strip().str.lower()
        
        valid_refs['total_goals'] = valid_refs['FTHG'].fillna(0) + valid_refs['FTAG'].fillna(0)
        valid_refs['total_cards'] = (valid_refs['HY'].fillna(0) + valid_refs['AY'].fillna(0) + 
                                   valid_refs['HR'].fillna(0) + valid_refs['AR'].fillna(0))
        valid_refs['is_over25'] = (valid_refs['total_goals'] > 2.5).astype(int)

        ref_grp = valid_refs.groupby('Referee').agg(
            matches=('Date', 'count'),goals=('total_goals', 'sum'),cards=('total_cards', 'sum'),over25=('is_over25', 'sum')
        )
        ref_grp = ref_grp[ref_grp['matches'] >= 5]
        
        ref_dict = {}
        for ref, row in ref_grp.iterrows():
            n = row['matches']
            ref_dict[ref] = json.dumps({
                'goals_pg': round(row['goals'] / n, 2),
                'cards_pg': round(row['cards'] / n, 2),
                'over25_rate': round(row['over25'] / n, 3),
                'strictness': round((row['cards']/n - 3.5) / 1.5, 3),
            })
            
        if ref_dict:
            r.hset('stavki:referee_profiles', mapping=ref_dict)
        logger.info(f"-> Cached target statistics for {len(ref_dict)} Referees")
        
        # Encodings
        if 'ref_encoded_goals' in df.columns:
            subset = df[['Referee', 'ref_encoded_goals', 'ref_encoded_cards']].dropna(subset=['Referee']).copy()
            subset['Referee'] = subset['Referee'].astype(str).str.strip().str.lower()
            subset = subset.drop_duplicates('Referee', keep='last').set_index('Referee')
            
            if 'ref_encoded_goals' in subset.columns:
                g = subset['ref_encoded_goals'].dropna().to_dict()
                if g: r.hset('stavki:ref_encoded_goals', mapping=g)
            if 'ref_encoded_cards' in subset.columns:
                c = subset['ref_encoded_cards'].dropna().to_dict()
                if c: r.hset('stavki:ref_encoded_cards', mapping=c)
            logger.info("-> Cached Neural Network Referee Embeddings")

    # Trigger a snapshot
    r.save()
    logger.info("✅ Redis BGSAVE triggered successfully. Memory serialization complete.")

if __name__ == "__main__":
    cache_live_state()
