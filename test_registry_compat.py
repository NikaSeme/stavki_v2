import pandas as pd
from pathlib import Path
from stavki.pipelines.daily import DailyPipeline
from stavki.data.schemas import Match

df = pd.read_parquet("data/features_enriched.parquet").head(10)
print(df.columns.tolist()[:20])

pipe = DailyPipeline()
matches = pipe._df_to_matches(df, is_historical=True)
print([m.is_completed for m in matches])
for m in matches:
    print(m.home_team.normalized_name, m.stats)
