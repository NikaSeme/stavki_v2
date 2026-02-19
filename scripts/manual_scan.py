#!/usr/bin/env python3
"""
Manual Pipeline Scanner
=======================
Runs the daily pipeline independently of the bot.
Does NOT affect the bot's scheduler or state.

Usage:
    python3 scripts/manual_scan.py
    python3 scripts/manual_scan.py --verbose
"""

import sys
import logging
from pathlib import Path

# Setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

# Logging
level = logging.DEBUG if "--verbose" in sys.argv else logging.INFO
logging.basicConfig(
    level=level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
# Silence noisy loggers
for name in ["httpx", "urllib3", "httpcore"]:
    logging.getLogger(name).setLevel(logging.WARNING)

logger = logging.getLogger("manual_scan")


def main():
    logger.info("🔍 Starting Manual Pipeline Scan...")

    from stavki.pipelines.daily import DailyPipeline

    pipeline = DailyPipeline()
    results = pipeline.run()

    # --- Print Results ---
    bets = results.get("bets", [])
    matches = results.get("matches_found", 0)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"SCAN COMPLETE: {matches} matches scanned, {len(bets)} value bets found")
    logger.info(f"{'='*60}")

    if not bets:
        logger.info("No value bets found.")
        # Still show model probabilities if available
        model_probs = results.get("model_probs", {})
        if model_probs:
            logger.info(f"\n📊 Model Probabilities for {len(model_probs)} matches:")
            for eid, probs in list(model_probs.items())[:10]:
                if isinstance(probs, dict):
                    # Might be nested {market: {outcome: prob}}
                    for market, outcomes in probs.items():
                        if isinstance(outcomes, dict):
                            h = outcomes.get("home", 0)
                            d = outcomes.get("draw", 0)
                            a = outcomes.get("away", 0)
                            logger.info(f"  Event {eid} [{market}]: H={h:.1%} D={d:.1%} A={a:.1%}")
                        else:
                            logger.info(f"  Event {eid}: {probs}")
                            break
        return

    # Sort by EV descending
    bets.sort(key=lambda b: b.get("ev", 0), reverse=True)

    for i, bet in enumerate(bets, 1):
        ev = bet.get("ev", 0)
        outcome = bet.get("outcome", "?")
        odds = bet.get("odds", 0)
        model_prob = bet.get("model_prob", 0)
        market_prob = bet.get("implied_prob", 0)
        match_name = bet.get("match", bet.get("home", "?") + " vs " + bet.get("away", "?"))
        league = bet.get("league", "?")
        bookmaker = bet.get("bookmaker", "?")

        edge_pct = (model_prob - market_prob) * 100 if market_prob else ev * 100

        logger.info(
            f"\n  #{i}  {match_name} ({league})"
            f"\n       Outcome: {outcome} @ {odds:.2f} ({bookmaker})"
            f"\n       Model: {model_prob:.1%}  Market: {market_prob:.1%}  Edge: {edge_pct:+.1f}%  EV: {ev:.1%}"
        )

    # Summary stats
    evs = [b.get("ev", 0) for b in bets]
    logger.info(f"\n📈 EV Summary: min={min(evs):.1%}, median={sorted(evs)[len(evs)//2]:.1%}, max={max(evs):.1%}")

    # Check for suspicious values
    high_ev = [b for b in bets if b.get("ev", 0) > 0.50]
    if high_ev:
        logger.warning(f"⚠️  {len(high_ev)} bets with EV > 50% — likely model miscalibration!")
    
    flat_probs = [b for b in bets if abs(b.get("model_prob", 0) - 0.333) < 0.05]
    if flat_probs:
        logger.warning(f"⚠️  {len(flat_probs)} bets with ~33% probability — model may still be flat!")


if __name__ == "__main__":
    main()
