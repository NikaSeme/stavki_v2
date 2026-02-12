import pandas as pd
df = pd.read_parquet("data/features_enriched.parquet")
print(f"Rows: {len(df)}")
print("Columns:", df.columns.tolist())
if not df.empty:
    valid_xg = df["xG_home"].notna().sum()
    print(f"Rows with xG: {valid_xg} / {len(df)}")
    print(df[["Date", "HomeTeam", "AwayTeam", "FTR", "xG_home"]].to_string())
