from stavki.pipelines.daily import DailyPipeline, PipelineConfig

pipeline = DailyPipeline(config=PipelineConfig(leagues=["soccer_epl"]))
print("Pipeline initialized")

try:
    hist_df = pipeline._load_history()
    print(f"Loaded {len(hist_df)} rows from history.")
    
    hist_matches = pipeline._df_to_matches(hist_df, is_historical=True)
    print(f"Successfully converted {len(hist_matches)} Match objects.")
    
    if len(hist_matches) == 0 and len(hist_df) > 0:
        for row in hist_df.head(1).itertuples(index=False):
            print("\nRow schema:", type(row))
            print("Columns available:", row._fields)
except Exception as e:
    import traceback
    traceback.print_exc()
