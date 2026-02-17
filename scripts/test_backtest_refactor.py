
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from stavki.backtesting.engine import BacktestEngine, BacktestConfig
from stavki.strategy.kelly import KellyStaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backtest_refactor():
    logger.info("Testing Backtest Refactor...")
    
    # Create synthetic data
    dates = [datetime.now() - timedelta(days=i) for i in range(10, 0, -1)]
    data = []
    
    for i, date in enumerate(dates):
        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "HomeTeam": f"Home_{i}",
            "AwayTeam": f"Away_{i}",
            "AvgOddsH": 2.0,
            "AvgOddsD": 3.5,
            "AvgOddsA": 4.0,
            "FTR": "H",  # Home win
            "League": "TestLeague"
        })
        
    df = pd.DataFrame(data)
    
    # Config
    config = BacktestConfig(
        min_ev=0.01,
        kelly_fraction=0.25,
        max_stake_pct=0.05,
        leagues=["TestLeague"],
    )

    
    engine = BacktestEngine(config=config)
    
    # Mock model probabilities (favoring Home slightly to trigger bets)
    # Market implied for Home (2.0) is 0.50.
    # We provide 0.55 to get +10% EV.
    model_probs = {}
    for i in range(len(df)):
        model_probs[i] = np.array([0.55, 0.25, 0.20])
        
    # Run backtest
    result = engine.run(df, model_probs=model_probs)
    
    logger.info("Backtest Result:")
    logger.info(f"Total Bets: {result.total_bets}")
    logger.info(f"Total Profit: {result.total_profit}")
    logger.info(f"ROI: {result.roi}")
    logger.info(f"Sharpe: {result.sharpe_ratio}")
    
    # Assertions
    assert result.total_bets > 0, "No bets placed"
    assert result.total_profit > 0, "Should be profitable with rigged probabilities"
    
    # Check staker state matches
    assert engine.staker.bankroll == 1000.0 + result.total_profit
    
    logger.info("Verification Successful!")

if __name__ == "__main__":
    test_backtest_refactor()
