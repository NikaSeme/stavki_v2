import csv

canonicals = set()
with open("data/mapping/canonical_teams.csv", "r") as f:
    r = csv.DictReader(f)
    for row in r:
        canonicals.add(row["canonical_name"])

print(f"Loaded {len(canonicals)} Master Teams.")

errors = 0
for source in ["odds_api.csv", "sportmonks.csv", "legacy_aliases.csv"]:
    try:
        with open(f"data/mapping/sources/{source}", "r") as f:
            r = csv.DictReader(f)
            for row in r:
                canon = row.get("canonical_name", "")
                if canon and canon not in canonicals:
                    print(f"[ERROR] {source}: {row['raw_name']} -> '{canon}' (NOT IN CANONICAL)")
                    errors += 1
    except FileNotFoundError:
        pass

if errors == 0:
    print("SUCCESS: All source mappings resolve to a valid canonical team.")
else:
    print(f"FAILED: Found {errors} mapping errors.")
