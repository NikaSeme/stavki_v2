"""
Diagnostic script: trace exactly WHY 0 value bets are found.
Dumps every intermediate value so we can see where the pipeline breaks.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
logging.basicConfig(level=logging.WARNING)

import pandas as pd
from stavki.pipelines.daily import DailyPipeline, PipelineConfig

def diagnose():
    config = PipelineConfig(min_ev=0.01, save_predictions=False)
    pipeline = DailyPipeline(config=config)
    pipeline._init_components()

    # 1. Odds
    odds = pipeline._fetch_odds()
    if odds is None or odds.empty:
        print("❌ No odds data")
        return

    print(f"{'='*60}")
    print(f"DIAGNOSTIC: Zero Value Bets Root Cause Analysis")
    print(f"{'='*60}")
    print(f"\n📊 Step 1: Odds Data")
    print(f"  Rows: {len(odds)}")
    print(f"  Columns: {list(odds.columns)}")
    print(f"  event_id dtype: {odds['event_id'].dtype}")
    print(f"  event_ids: {odds['event_id'].tolist()}")

    # 2. Matches
    matches_df = pipeline._extract_matches(odds)
    pipeline._enrich_matches(matches_df)
    print(f"\n📊 Step 2: Matches")
    print(f"  Count: {len(matches_df)}")
    print(f"  event_id dtype: {matches_df['event_id'].dtype}")
    print(f"  event_ids: {matches_df['event_id'].tolist()}")

    # 3. Features
    features_df = pipeline._build_features(matches_df, odds)
    print(f"\n📊 Step 3: Features")
    print(f"  Shape: {features_df.shape}")

    # 4. Model Predictions
    probs = pipeline._get_predictions(matches_df, features_df)
    print(f"\n📊 Step 4: Model Predictions")
    print(f"  match_ids in probs: {list(probs.keys())}")
    for mid, pdict in probs.items():
        print(f"  {mid} (type={type(mid).__name__}):")
        for k, v in pdict.items():
            print(f"    {k}: {v:.4f}")

    # 5. Best Prices
    best_prices = pipeline._select_best_prices(odds)
    print(f"\n📊 Step 5: Best Prices")
    print(f"  Shape: {best_prices.shape}")
    print(f"  Columns: {list(best_prices.columns)}")
    if "event_id" in best_prices.columns:
        print(f"  event_id dtype: {best_prices['event_id'].dtype}")
        print(f"  event_ids: {best_prices['event_id'].unique().tolist()}")
    if "outcome_name" in best_prices.columns:
        print(f"  outcome_names: {best_prices['outcome_name'].unique().tolist()}")
    if "outcome_price" in best_prices.columns:
        print(f"  outcome_prices: {best_prices['outcome_price'].tolist()}")

    # 6. Market Probs
    market_probs = pipeline._compute_market_probs(best_prices)
    print(f"\n📊 Step 6: Market Probs (no-vig)")
    for mid, mp in market_probs.items():
        print(f"  {mid} (type={type(mid).__name__}):")
        for k, v in mp.items():
            print(f"    {k}: {v:.4f}")

    # 7. Value Bet Detection - MANUAL TRACE
    print(f"\n{'='*60}")
    print(f"📊 Step 7: MANUAL VALUE BET TRACE")
    print(f"{'='*60}")

    for _, match in matches_df.iterrows():
        event_id = match["event_id"]
        home = match.get("home_team", "Home")
        away = match.get("away_team", "Away")
        league = match.get("league", "unknown")

        print(f"\n  Match: {home} vs {away}")
        print(f"  event_id: {event_id} (type={type(event_id).__name__})")

        # Check model probs
        if event_id not in probs:
            # Try type coercion
            str_id = str(event_id)
            int_id = None
            try:
                int_id = int(event_id)
            except:
                pass

            found = False
            for k in probs.keys():
                if str(k) == str_id:
                    print(f"  ⚠️ TYPE MISMATCH: event_id={event_id} ({type(event_id).__name__}) vs probs key={k} ({type(k).__name__})")
                    found = True
                    break
            if not found:
                print(f"  ❌ event_id NOT in model_probs (keys: {list(probs.keys())})")
            continue

        event_model_probs = probs[event_id]
        event_market_probs = market_probs.get(event_id, {})

        if not event_market_probs:
            # Try type coercion for market_probs too
            for k in market_probs.keys():
                if str(k) == str(event_id):
                    print(f"  ⚠️ MARKET PROBS TYPE MISMATCH: event_id type={type(event_id).__name__}, key type={type(k).__name__}")
                    event_market_probs = market_probs[k]
                    break

        print(f"  Market probs: {event_market_probs}")
        print(f"  Model probs: {event_model_probs}")

        # Get best prices for this event  
        event_prices = best_prices[best_prices["event_id"] == event_id]
        if event_prices.empty:
            # Try string comparison
            event_prices = best_prices[best_prices["event_id"].astype(str) == str(event_id)]
            if not event_prices.empty:
                print(f"  ⚠️ BEST PRICES TYPE MISMATCH (fixed with str cast)")

        if event_prices.empty:
            print(f"  ❌ No prices found for event_id={event_id}")
            continue

        print(f"  Prices rows: {len(event_prices)}")

        for _, price_row in event_prices.iterrows():
            outcome = price_row.get("outcome_name")
            odds_val = price_row.get("outcome_price", 2.0)

            p_model = event_model_probs.get(outcome)
            if p_model is None:
                p_model = event_model_probs.get(outcome.lower()) or event_model_probs.get(outcome.title())

            if p_model is None:
                print(f"    {outcome}: ❌ No model prob (available keys: {list(event_model_probs.keys())})")
                continue

            p_market = event_market_probs.get(outcome, 1.0 / odds_val)

            # Blender
            blender = pipeline._blender
            tier = blender.router._get_tier(league)
            alpha = blender.alphas.get(tier, 0.50)
            p_blended = blender.blend(p_model, p_market, league)

            # EV
            ev = p_blended * odds_val - 1
            edge = p_blended - (1.0 / odds_val)

            # What would EV be WITHOUT blending (raw model)?
            ev_raw = p_model * odds_val - 1

            print(f"    {outcome}:")
            print(f"      odds={odds_val:.3f}")
            print(f"      p_model={p_model:.4f}")
            print(f"      p_market={p_market:.4f}")
            print(f"      league='{league}' → tier={tier} → alpha={alpha}")
            print(f"      p_blended={p_blended:.4f} (={alpha}*{p_model:.4f} + {1-alpha}*{p_market:.4f})")
            print(f"      EV (blended) = {ev:.4f} ({ev:.2%})")
            print(f"      EV (raw model) = {ev_raw:.4f} ({ev_raw:.2%})")
            print(f"      min_ev threshold = {config.min_ev}")
            if ev >= config.min_ev:
                print(f"      ✅ PASSES EV filter")
            else:
                print(f"      ❌ FAILS EV filter (need {config.min_ev}, got {ev:.4f})")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY OF ISSUES")
    print(f"{'='*60}")
    print(f"  1. Only 1X2 market odds flow to value detection (no O/U, no BTTS)")
    print(f"     → Model predicts O/U and BTTS but there are no odds to compare")
    print(f"  2. Blender alpha for tier1 (EPL) = {pipeline._blender.alphas.get('tier1', 'N/A')}")
    print(f"     → Only {pipeline._blender.alphas.get('tier1', 0.3)*100:.0f}% model weight")
    print(f"     → Even if model finds edge, blender dilutes it toward market")
    print(f"  3. Check type mismatches above (event_id str vs int)")

if __name__ == "__main__":
    diagnose()
