from stavki.pipelines import DailyPipeline, PipelineConfig
import pandas as pd
import json

config = PipelineConfig(leagues=["soccer_spain_la_liga"], min_ev=0.0)
pipeline = DailyPipeline(config=config, bankroll=1000)
bets = pipeline.run()

print(f"Found {len(bets)} total bets.")
for bet in bets:
    if "Barcelona" in f"{bet.home_team} vs {bet.away_team}" or bet.ev > 1.5:
        print(json.dumps(bet.to_dict(), indent=2))
