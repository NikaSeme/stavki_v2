
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from stavki.config import PROJECT_ROOT, DATA_DIR
from stavki.models.training.trainer import ModelTrainer
from stavki.models.base import Market
from stavki.backtesting.engine import BacktestEngine, BacktestConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Comprehensive Backtest...")
    
    # 1. Load Data
    data_path = DATA_DIR / "features_full.parquet"
    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        return
        
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Ensure consistent match_id
    if "match_id" not in df.columns:
        logger.info("Generating match_ids...")
        df["match_id"] = df.apply(lambda r: f"{r['HomeTeam']}_vs_{r['AwayTeam']}_{r.name}", axis=1)
    
    # Sort by date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=True)
        df = df.sort_values("Date")
    
    # 2. Load Models
    models_dir = PROJECT_ROOT / "models"
    if not models_dir.exists():
        logger.error(f"Models directory not found at {models_dir}. Run training first.")
        return
        
    trainer = ModelTrainer(models_dir=models_dir)
    trainer.load_models()
    
    if not trainer.ensemble:
        logger.error("Ensemble not loaded! Check if ensemble_weights.json exists.")
        return
        
    # 3. Define Holdout Set (Last 20%)
    n = len(df)
    holdout_size = int(n * 0.20)
    test_df = df.iloc[-holdout_size:].copy()
    
    logger.info(f"Running backtest on last {holdout_size} matches ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    # 4. Generate Predictions
    logger.info("Generating ensemble predictions...")
    # Clean dataframe for prediction (handle potential categorical mismatches if any)
    # The models should handle normalization, but we ensure basic types
    predictions = trainer.ensemble.predict(test_df)
    
    # Map predictions to index for BacktestEngine
    # BacktestEngine expects {index: np.array([p_home, p_draw, p_away])}
    pred_map = {p.match_id: p for p in predictions if p.market == Market.MATCH_WINNER}
    
    model_probs = {}
    found_count = 0
    
    for idx, row in test_df.iterrows():
        match_id = row.get("match_id")
        if match_id in pred_map:
            p = pred_map[match_id].probabilities
            # Order: Home, Draw, Away
            probs = np.array([p.get('home', 0.0), p.get('draw', 0.0), p.get('away', 0.0)])
            # Normalize just in case
            probs = probs / (probs.sum() + 1e-9)
            
            model_probs[idx] = probs
            found_count += 1
            
    logger.info(f"Generated probabilities for {found_count}/{len(test_df)} matches")
    
    # 5. Run Backtest
    # aggressive kelly for testing potential
    config = BacktestConfig(
        leagues=[], # All leagues
        min_ev=0.03, # 3% EV threshold
        min_edge=0.0, # Any edge
        kelly_fraction=0.1, # Conservative kelly
        max_stake_pct=0.05,
    )
    
    engine = BacktestEngine(config)
    result = engine.run(test_df, model_probs=model_probs)
    
    # 6. Report Results
    logger.info("\n" + "="*40)
    logger.info(" BACKTEST RESULTS ")
    logger.info("="*40)
    logger.info(f"Total Bets:      {result.total_bets}")
    logger.info(f"Win Rate:        {result.win_rate:.2%}")
    logger.info(f"Total Staked:    {result.total_stake:.2f}u")
    logger.info(f"Total Profit:    {result.total_profit:.2f}u")
    logger.info(f"ROI:             {result.roi:.2%}")
    logger.info(f"Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    logger.info(f"Max Drawdown:    {result.max_drawdown:.2%}")
    logger.info("="*40)
    
    # Per-league stats
    logger.info("\nPer-League Performance:")
    sorted_leagues = sorted(result.league_results.items(), key=lambda x: x[1]['profit'], reverse=True)
    for league, stats in sorted_leagues:
        if stats['bets'] > 0:
            logger.info(f"{league:<15} | Bets: {stats['bets']:<4} | ROI: {stats['roi']:>7.2%} | Profit: {stats['profit']:>7.2f}u")

if __name__ == "__main__":
    main()
