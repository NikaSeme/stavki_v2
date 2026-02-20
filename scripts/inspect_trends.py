
import sys
import json
import gzip
from pathlib import Path

def inspect_trends(path):
    print(f"🕵️♂️ Inspecting Trends/Comments in {path}")
    
    with gzip.open(path, 'rt', encoding='UTF-8') as f:
        data = json.load(f)
        
    # 1. Trends
    trends = data.get('trends', [])
    print(f"\n1. Trends (Count: {len(trends)})")
    if trends:
        # Sample first few
        for t in trends[:5]:
            print(f"   - {t}")
            
    # 2. Comments
    comments = data.get('comments', [])
    print(f"\n2. Comments (Count: {len(comments)})")
    if comments:
        # Sample interesting ones (VAR?)
        for c in comments:
            txt = c.get('comment', '').lower()
            if any(k in txt for k in ['var', 'penalty', 'red card', 'injury', 'substitution', 'own goal']):
                print(f"   - {c.get('minute')}: {c.get('comment')}")
        # Print first 3 if no interesting ones
        if not any(k in c.get('comment', '').lower() for k in ['var', 'penalty', 'red card', 'injury'] for c in comments):
             for c in comments[:3]:
                 print(f"   - {c.get('minute')}: {c.get('comment')}")

if __name__ == "__main__":
    # Find a file
    import glob
    files = glob.glob("data/raw/fixtures/*/*.json.gz")
    if files:
        files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        inspect_trends(files[0])
