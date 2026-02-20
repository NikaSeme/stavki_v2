"""
Real xG Feature Builder (Tier 1).

Replaces Synthetic xG with actual xG data from SportMonks.
Reads from local backfilled storage for historical matches to ensure
consistency between training and inference.

Features produced:
  - xg_home / xg_away (Real SportMonks xG)
  - xg_diff
  - xg_efficiency_home / _away (Goals - xG)
"""

import json
import logging
from typing import Dict, Optional, List
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from stavki.config import PROJECT_ROOT
from stavki.data.schemas import Match

logger = logging.getLogger(__name__)

STORAGE_PATH = PROJECT_ROOT / "stavki" / "data" / "storage" / "sportmonks_historical.jsonl"

class RealXGBuilder:
    """
    Compute features from Real SportMonks xG data.
    
    For historical training: Reads from `sportmonks_historical.jsonl`.
    For live inference: Uses `match.stats` directly.
    """
    
    name = "real_xg"
    
    def __init__(self, rolling_window: int = 10):
        self.rolling_window = rolling_window
        self._historical_data: Dict[str, Dict] = {} # Key: date_home_away -> stats
        self._team_xg_history: Dict[str, List[Dict]] = defaultdict(list)
        self._is_fitted = False
        
        # Load historical storage if exists
        self._load_storage()

    def _load_storage(self):
        """Load backfilled data into memory index."""
        if not STORAGE_PATH.exists():
            logger.warning(f"RealXGBuilder: No historical storage found at {STORAGE_PATH}")
            return
            
        try:
            with open(STORAGE_PATH, 'r') as f:
                count = 0
                for line in f:
                    try:
                        record = json.loads(line)
                        # Create composite key: date_home_away
                        key = f"{record['date']}_{record['home_team']}_{record['away_team']}"
                        
                        # Extract relevant stats to save memory
                        # Structure: record['sm_data']['stats']['data'] -> list of 2 teams
                        stats_data = {}
                        sm_data = record.get('sm_data', {})
                        
                        # Handle different API structures if needed (v3 is nested)
                        # v3: output of backfill is the WHOLE fixture object
                        # We need to find the stats inside it
                        
                        # Flatten data
                        stats = sm_data.get('statistics', [])
                        if not stats:
                            continue
                            
                        # Parse stats (SportMonks returns list of dicts, one per team)
                        for team_stat in stats:
                            # Identify if home or away? 
                            # SportMonks v3 statistics usually include team_id or type
                            # We might need team_ids from the fixture to map correctly
                            # For now, let's assume we can map by team_id
                            
                            # Wait, backfill saves the whole fixture.
                            # We need participants to map IDs to Home/Away
                            participants = sm_data.get('participants', [])
                            home_id = next((p['id'] for p in participants if p['meta']['location'] == 'home'), None)
                            away_id = next((p['id'] for p in participants if p['meta']['location'] == 'away'), None)
                            
                            stat_team_id = team_stat.get('participant_id')
                            location = None
                            if stat_team_id == home_id: location = 'home'
                            elif stat_team_id == away_id: location = 'away'
                            
                            if location:
                                # Extract xG. SM v3 xG is often in 'statistics' -> type='expected_goals'
                                # But wait, the python client `_parse_statistics` handles this mapping.
                                # The backfill saves the RAW json. We need to parse it or rely on the client's logic?
                                # Let's parse raw JSON here to be independent.
                                
                                # SM v3 stats is a list of objects like: 
                                # { "type": {"id": 12, "name": "Big Chances Created", ...}, "value": 2 }
                                # OR if using the `include=statistics`, it might be a list of team stats.
                                
                                # Actually, checking `sportmonks.py`: 
                                # `get_fixture_stats` calls `includes=["statistics"]`.
                                # The backfill script calls `includes=["statistics", ...]`
                                
                                # Let's assume standard v3 format:
                                # statistics: [ { team_id: X, type: "Session", stats: [ {label: "Goals", value: 1}, ... ] } ]
                                # Wait, SportMonks v3 structure is complex.
                                # Let's simplify: Just look for "Expected Goals" in the data.
                                pass 
                        
                        # To avoid complex parsing here, let's trust the backfill script 
                        # to save the RAW response. We will need robust parsing here.
                        # Actually, better strategy: 
                        # The `SportMonksClient` already has `_parse_statistics`.
                        # We should potentially reuse that logic.
                        
                        # But we can't easily import the client instance methods without a client.
                        # Let's implement a lightweight parser here.
                        
                        # ... (Parsing logic implemented below in `_parse_raw_stats`) ...
                        
                        parsed = self._parse_raw_stats(sm_data)
                        if parsed:
                            self._historical_data[key] = parsed
                            count += 1
                            
                    except json.JSONDecodeError:
                        continue
                        
            logger.info(f"RealXGBuilder: Loaded {count} historical matches with xG")
            
        except Exception as e:
            logger.error(f"RealXGBuilder: Failed to load storage: {e}")

    def _parse_raw_stats(self, sm_data: Dict) -> Optional[Dict]:
        """Parse raw SportMonks fixture JSON to extract xG or compute proxy."""
        # Get Team IDs
        participants = sm_data.get('participants', [])
        home_id = next((p['id'] for p in participants if p.get('meta', {}).get('location') == 'home'), None)
        away_id = next((p['id'] for p in participants if p.get('meta', {}).get('location') == 'away'), None)
        
        if not home_id or not away_id:
            return None
            
        stats = sm_data.get('statistics', [])
        if not stats:
            return None

        # Container for accumulated stats
        # Keys: home_xg, away_xg, home_big_chances, away_big_chances, etc.
        data = defaultdict(lambda: defaultdict(float))
        
        # Mapping IDs to internal keys
        # Based on investigation:
        # 580: Big Chances Created (Best proxy for high xG)
        # 49: Shots Insidebox
        # 50: Shots Outsidebox
        # Penalties? Usually included in shots?
        
        TYPE_MAP = {
            "Expected Goals": "xg",
            "xG": "xg",
            580: "big_chances",
            49: "shots_inside",
            50: "shots_outside",
            86: "sot", # Shots on target
            41: "swoff", # Shots off target
            42: "shots_total"
        }
        
        found_xg = False
        
        for item in stats:
            team_id = item.get('participant_id')
            if team_id not in [home_id, away_id]:
                continue
                
            side = 'home' if team_id == home_id else 'away'
            
            # Determine type
            type_obj = item.get('type')
            type_id = item.get('type_id')
            stat_key = None
            
            # 1. Check Name
            if isinstance(type_obj, dict):
                name = type_obj.get('name')
                if name in TYPE_MAP:
                    stat_key = TYPE_MAP[name]
            elif isinstance(type_obj, str) and type_obj in TYPE_MAP:
                stat_key = TYPE_MAP[type_obj]
                
            # 2. Check ID (Fallback)
            if not stat_key and type_id in TYPE_MAP:
                stat_key = TYPE_MAP[type_id]
                
            if stat_key:
                # Value extraction:
                # V3 often puts value in `data` object: {"data": {"value": 2}}
                # Or sometimes at top level `value`: 2
                
                raw_val = item.get('value')
                if raw_val is None:
                     raw_val = item.get('data', {}).get('value')
                     
                # Handle complex value (e.g. {"total": 5, "missed": 2})
                if isinstance(raw_val, dict):
                    val = raw_val.get('total')
                else:
                    val = raw_val
                    
                # Handle None
                if val is None: val = 0.0
                data[side][stat_key] = float(val)
                
                if stat_key == 'xg':
                    found_xg = True
                    
        # Compute Proxy xG if missing
        result = {}
        for side in ['home', 'away']:
            if 'xg' in data[side] and data[side]['xg'] > 0:
                result[f"{side}_xg"] = data[side]['xg']
            else:
                # Proxy Formula
                # xG ≈ (Big Chances * 0.45) + (Shots Inside * 0.08) + (Shots Outside * 0.03)
                # Note: Big Chances usually overlap with shots inside. 
                # If Big Chance, it accounts for ~0.45.
                # If Shot Inside (Not Big), it's ~0.08.
                # But we don't know which shots are big chances.
                # Assumption: Big Chances are a SUBSET of Shots Inside.
                # So: xG = (Big Chances * 0.40) + ((Shots Inside - Big Chances) * 0.07) + (Shots Outside * 0.03)
                
                bc = data[side]['big_chances']
                inside = max(0, data[side]['shots_inside'] - bc)
                outside = data[side]['shots_outside']
                
                proxy_xg = (bc * 0.45) + (inside * 0.08) + (outside * 0.03)
                result[f"{side}_xg"] = round(proxy_xg, 2)
                
                # If simple shots total exists but no detail?
                if proxy_xg == 0 and data[side]['shots_total'] > 0:
                    # Fallback coarse: 0.10 per shot
                     result[f"{side}_xg"] = round(data[side]['shots_total'] * 0.10, 2)

        return result

    def fit(self, matches: List[Match]) -> None:
        """Build rolling history from historical matches."""
        self._team_xg_history.clear()
        
        sorted_matches = sorted(matches, key=lambda x: x.commence_time)
        
        for m in sorted_matches:
            # Try to find stats
            stats = None
            
            # 1. Check if match object already has rich stats (e.g. from live loader)
            if m.stats and (m.stats.home_xg is not None or m.stats.away_xg is not None):
                stats = {
                    'home_xg': m.stats.home_xg or 0.0,
                    'away_xg': m.stats.away_xg or 0.0
                }
            
            # 2. Key lookup in backfilled storage
            else:
                key = f"{m.commence_time.strftime('%Y-%m-%d')}_{m.home_team.normalized_name}_{m.away_team.normalized_name}"
                stats = self._historical_data.get(key)
            
            if stats:
                # Add to history
                self._team_xg_history[m.home_team.normalized_name].append({
                    "xg": stats['home_xg'],
                    "goals": m.home_score or 0,
                    "date": m.commence_time
                })
                self._team_xg_history[m.away_team.normalized_name].append({
                    "xg": stats['away_xg'],
                    "goals": m.away_score or 0,
                    "date": m.commence_time
                })
                
                # Maintain window
                for team in [m.home_team.normalized_name, m.away_team.normalized_name]:
                    if len(self._team_xg_history[team]) > self.rolling_window * 2:
                        self._team_xg_history[team] = self._team_xg_history[team][-self.rolling_window * 2:]
        
        self._is_fitted = True
        logger.info(f"RealXGBuilder: Fitted on {len(sorted_matches)} matches.")

    def get_features(
        self,
        match: Optional[Match] = None,
        as_of: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Compute real xG features."""
        # Defaults
        defaults = {
            "xg_home": 1.35, # League Avg approx
            "xg_away": 1.15,
            "xg_diff": 0.20,
            "xg_efficiency_home": 0.0,
            "xg_efficiency_away": 0.0,
        }
        
        if not match:
            return defaults
            
        features = {}
        ref_time = as_of or match.commence_time
        
        # 1. Rolling Average xG (Form)
        for side, team_name in [
            ("home", match.home_team.normalized_name),
            ("away", match.away_team.normalized_name),
        ]:
            history = self._team_xg_history.get(team_name, [])
            if ref_time:
                history = [h for h in history if h["date"] < ref_time]
            
            recent = history[-self.rolling_window:]
            
            if recent:
                avg_xg = sum(h["xg"] for h in recent) / len(recent)
                avg_goals = sum(h["goals"] for h in recent) / len(recent)
                efficiency = round(avg_goals - avg_xg, 3) # + means scoring more than expected (lucky/skill)
            else:
                avg_xg = defaults[f"xg_{side}"]
                efficiency = 0.0
                
            features[f"xg_{side}"] = round(avg_xg, 3)
            features[f"xg_efficiency_{side}"] = efficiency

        features["xg_diff"] = round(features["xg_home"] - features["xg_away"], 3)
        return features
