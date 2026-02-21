import logging
import json
from stavki.pipelines.daily import DailyPipeline, PipelineConfig

logging.basicConfig(level=logging.ERROR)

config = PipelineConfig(leagues=["soccer_epl"], min_ev=-1.0)
pipeline = DailyPipeline(config=config)

print("\n--- RUNNING FULL PIPELINE EXTRACT & INFERENCE ---")
odds_df = pipeline._fetch_odds()
matches_df = pipeline._extract_matches(odds_df)
pipeline._enrich_matches(matches_df)

features_df = pipeline._build_features(matches_df, odds_df)
probs = pipeline._get_predictions(matches_df, features_df)

print(f"\nSUCCESS: Generated {len(probs)} prediction batches (grouped by event_id).")
for k, v in list(probs.items())[:3]:
    print(f"  Match {k}: {len(v)} markets predicted.")
