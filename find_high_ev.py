import json
import glob
import os

for f in glob.glob("outputs/bets/*.json"):
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            for bet in data.get("bets", []):
                if bet.get("ev", 0) > 2.0:
                    print(f"High EV found in {f}:")
                    print(json.dumps(bet, indent=2))
    except Exception as e:
        pass
