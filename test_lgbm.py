import sys
import warnings
warnings.filterwarnings('ignore')

try:
    from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
    model = LightGBMModel.load('models/LightGBM_1X2.pkl')
    features = model.features
    print(f"Num Features Expected: {len(features)}")
    
    from stavki.pipelines.daily import DailyPipeline, PipelineConfig
    config = PipelineConfig(leagues=["soccer_epl"], min_ev=-1.0)
    pipeline = DailyPipeline(config=config)
    odds_df = pipeline._fetch_odds()
    matches_df = pipeline._extract_matches(odds_df)
    pipeline._enrich_matches(matches_df)
    features_df = pipeline._build_features(matches_df, odds_df)
    available = list(features_df.columns)
    
    missing = [f for f in features if f not in available]
    print(f"Missing ({len(missing)}):\n{missing[:40]}")
except Exception as e:
    import traceback
    traceback.print_exc()
