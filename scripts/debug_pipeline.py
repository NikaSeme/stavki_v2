"""
Debug Pipeline — Step-by-step trace of the daily value-bet pipeline.

Runs each pipeline stage individually with verbose output to make it easy
to spot exactly where data gets lost or mis-shaped.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import pandas as pd
from stavki.pipelines.daily import DailyPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger(__name__)


def debug():
    print("🚀 Starting Debug Pipeline...\n")
    config = PipelineConfig(min_ev=0.01, save_predictions=False)
    pipeline = DailyPipeline(config=config)
    pipeline._init_components()

    # ── Step 1: Fetch Odds ──────────────────────────────────────────────
    print("Step 1: Fetching Odds...")
    odds = pipeline._fetch_odds()
    if odds is None or odds.empty:
        print("❌ Odds fetch returned empty/None!")
        return

    print(f"  ✅ {len(odds)} odds rows fetched")
    print(f"     Columns: {list(odds.columns)}\n")

    # ── Step 2: Extract & enrich matches ────────────────────────────────
    print("Step 2: Extracting matches...")
    matches_df = pipeline._extract_matches(odds)
    pipeline._enrich_matches(matches_df)
    print(f"  ✅ {len(matches_df)} unique matches\n")

    # ── Step 3: Build features ──────────────────────────────────────────
    print("Step 3: Building features...")
    features_df = pipeline._build_features(matches_df, odds)
    print(f"  ✅ Features shape: {features_df.shape}")

    # Quick feature audit
    key_features = ["elo_home", "form_home_pts", "form_home_gf", "AvgH", "AvgD", "AvgA"]
    present = [f for f in key_features if f in features_df.columns]
    missing = [f for f in key_features if f not in features_df.columns]
    if present:
        print(f"     Key features present: {present}")
    if missing:
        print(f"     ⚠️  Missing features: {missing}")
    print()

    # ── Step 4: Model predictions ───────────────────────────────────────
    print("Step 4: Running model predictions...")
    probs = pipeline._get_predictions(matches_df, features_df)
    print(f"  ✅ {len(probs)} matches predicted")

    for match_id, markets in probs.items():
        print(f"     {match_id}:")
        for market, outcomes in markets.items():
            fmt = "  ".join(f"{k}={v:.3f}" for k, v in outcomes.items())
            print(f"       [{market}]  {fmt}")
    print()

    # ── Step 5: Best prices & market probs ──────────────────────────────
    print("Step 5: Best prices & market probs...")
    best_prices = pipeline._select_best_prices(odds)
    market_probs = pipeline._compute_market_probs(best_prices)
    print(f"  ✅ Best prices: {len(best_prices)} rows")
    print(f"  ✅ Market probs: {len(market_probs)} events\n")

    # ── Step 6: Finding value ───────────────────────────────────────────
    print("Step 6: Finding value bets...")
    candidates = pipeline._find_value_bets(
        matches_df, probs, market_probs, best_prices,
    )
    print(f"  ✅ {len(candidates)} value bets found\n")

    if not candidates:
        print("  No candidates — dumping EV trace per outcome:\n")
        _trace_ev(pipeline, matches_df, probs, market_probs, best_prices, config)
        return

    for i, c in enumerate(candidates[:10], 1):
        print(
            f"  #{i}  {c.home_team} vs {c.away_team}  |  {c.market} → {c.selection}  "
            f"@ {c.odds:.2f}  |  EV={c.ev:+.2%}  "
            f"(model={c.model_prob:.3f}  market={c.market_prob:.3f}  "
            f"blended={c.blended_prob:.3f})"
        )


def _trace_ev(pipeline, matches_df, probs, market_probs, best_prices, config):
    """When 0 candidates found, trace every EV computation to show why."""
    for _, match in matches_df.iterrows():
        eid = match["event_id"]
        home = match.get("home_team", "?")
        away = match.get("away_team", "?")
        league = match.get("league", "unknown")

        event_markets = probs.get(eid, {})
        event_mkt_probs = market_probs.get(eid, {})
        event_prices = best_prices[best_prices["event_id"] == eid]

        print(f"  {home} vs {away}  (event_id={eid})")

        if not event_markets:
            print("    ❌ No model predictions for this event\n")
            continue

        for _, pr in event_prices.iterrows():
            outcome = pr.get("outcome_name")
            odds_val = pr.get("outcome_price", 0)
            market_key = pipeline.OUTCOME_TO_MARKET.get(outcome)
            p_model = event_markets.get(market_key, {}).get(outcome)
            p_market = event_mkt_probs.get(outcome, 1.0 / odds_val if odds_val > 1 else 0)

            if p_model is None:
                print(f"    {outcome}: no model prob (market_key={market_key})")
                continue

            tier = pipeline._blender.router._get_tier(league)
            alpha = pipeline._blender.alphas.get(tier, 0.5)
            p_blended = pipeline._blender.blend(p_model, p_market, league)
            ev = p_blended * odds_val - 1

            status = "✅ PASS" if ev >= config.min_ev else "❌ FAIL"
            print(
                f"    {outcome}: odds={odds_val:.3f}  p_model={p_model:.4f}  "
                f"p_market={p_market:.4f}  α={alpha}  "
                f"p_blend={p_blended:.4f}  EV={ev:+.4f}  {status}"
            )
        print()


if __name__ == "__main__":
    debug()
