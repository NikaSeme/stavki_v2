import json
import glob
import os

count = 0
found = False
print("Scanning JSON files...")
for f in glob.glob("outputs/bets/*.json"):
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            count += 1
            for bet in data.get("bets", []):
                if bet.get("ev", 0) > 1.5:
                    print(f"HIGH EV in {f}: {bet['match']} - {bet['selection']} - {bet['ev']}")
                    found = True
                if "barcelona" in bet.get("match", "").lower():
                    print(f"BARCELONA MATCH in {f}: {bet['match']} - {bet['selection']} - ew: {bet['ev']}")
                    found = True
    except Exception as e:
        print(f"Error {f}: {e}")

print(f"Processed {count} files.")
if not found:
    print("NO high EV bets found.")
