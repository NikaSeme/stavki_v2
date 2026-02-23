import csv
from pathlib import Path

ODDS_CSV_PATH = Path("data/mapping/sources/odds_api.csv")

manual_mappings = {
    "Brighton and Hove Albion": "Brighton",
    "Celta Vigo": "Celta",
    "Oviedo": "Oviedo",
    "Real Betis": "Betis",
    "FC St. Pauli": "St Pauli",
    "FSV Mainz 05": "Mainz",
    "Pisa": "Pisa",
    "Como": "Como",
    "Hellas Verona": "Verona",
    "Paris Saint Germain": "Paris SG",
    "Wrexham AFC": "Wrexham",
    "Portsmouth": "Portsmouth",
    "Preston North End": "Preston",
    "Queens Park Rangers": "QPR",
    "Oxford United": "Oxford"
}

matches = []
with open(ODDS_CSV_PATH, "r") as f:
    existing = {row["raw_name"] for row in csv.DictReader(f)}

for raw, can in manual_mappings.items():
    if raw not in existing:
        matches.append({"raw_name": raw, "canonical_name": can})

if matches:
    with open(ODDS_CSV_PATH, "a") as f:
        writer = csv.DictWriter(f, fieldnames=["raw_name", "canonical_name"])
        for m in matches:
            writer.writerow(m)
    print(f"Appended {len(matches)} manual mappings!")
else:
    print("No new mappings to append.")
