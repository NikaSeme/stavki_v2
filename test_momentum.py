import logging
import sys

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

from stavki.data.collectors.momentum_odds import MomentumOddsCollector

collector = MomentumOddsCollector()
captured = collector.capture()
print(f"Test complete. Captured odds for {captured} upcoming matches.")
