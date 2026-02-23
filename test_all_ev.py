import json
import glob
print("Scanning for high EV bets...")
count = 0
for f in glob.glob("outputs/bets/*.json"):
    with open(f, 'r') as file:
        data = json.load(file)
        count += 1
        for bet in data.get("bets", []):
            if bet.get("ev", 0) > 0.2:
                print(f"File {f}: {bet['match']} - {bet['selection']} - ev: {bet['ev']}")

print(f"Successfully processed {count} JSON files.")
