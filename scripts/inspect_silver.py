
import pandas as pd
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from stavki.config import PROJECT_ROOT

def inspect():
    print("🕵️♂️ Inspecting Silver Data")
    
    # 1. Players
    p_path = PROJECT_ROOT / "data" / "processed" / "players" / "player_stats_silver.parquet"
    if p_path.exists():
        df = pd.read_parquet(p_path)
        print(f"\n1. Players: {df.shape}")
        print(df.head())
        print("\n   Stat Counts:")
        print(df[['rating', 'minutes', 'goals', 'shots']].count())
        print("\n   Sample Ratings:")
        print(df['rating'].dropna().head())
    else:
        print("   ❌ No players file found.")
        
    # 2. Injuries
    i_path = PROJECT_ROOT / "data" / "processed" / "injuries" / "injuries_silver.parquet"
    if i_path.exists():
        df = pd.read_parquet(i_path)
        print(f"\n2. Injuries: {df.shape}")
        print(df[['player_id', 'team_id', 'reason', 'start_date']].head())
        print("\n   Top Reasons:")
        print(df['reason'].value_counts().head())
    else:
        print("   ❌ No injuries file found.")

if __name__ == "__main__":
    inspect()
