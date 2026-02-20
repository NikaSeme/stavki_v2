
import pandas as pd
import json
from pathlib import Path

def inspect():
    print("🕵️♂️ Inspecting Data Sources...")
    
    # 1. Check Parquet
    try:
        df = pd.read_parquet('data/features_enriched.parquet')
        print(f"\n1. features_enriched.parquet: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        if 'match_id' in df.columns:
            print(f"   ✅ match_id found! Sample: {df['match_id'].iloc[0]}")
        elif 'id' in df.columns:
             print(f"   ✅ id found! Sample: {df['id'].iloc[0]}")
        else:
             print("   ❌ No obvious ID column found.")
    except Exception as e:
        print(f"   ❌ Failed to read parquet: {e}")

    # 2. Check JSON Map
    try:
        with open('data/fixture_id_map.json') as f:
            data = json.load(f)
        print(f"\n2. fixture_id_map.json Type: {type(data)}")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())[:5]}")
            # Peek at first key
            k1 = list(data.keys())[0]
            print(f"   Sample [{k1}]: {data[k1]}")
        elif isinstance(data, list):
            print(f"   List length: {len(data)}")
            print(f"   Sample: {data[0]}")
    except Exception as e:
        print(f"   ❌ Failed to read JSON: {e}")

if __name__ == "__main__":
    inspect()
