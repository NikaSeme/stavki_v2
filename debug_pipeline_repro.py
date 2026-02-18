
import logging
import sys
import os
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DebugPipeline")

# Add project root to path
sys.path.insert(0, os.getcwd())

from stavki.pipelines.daily import DailyPipeline, PipelineConfig

def run_debug():
    logger.info("Starting Debug Run")
    
    # Configure pipeline for maximum discovery
    config = PipelineConfig(
        min_ev=-1.0,  # Negative EV to guarantee we see bets (if prediction works)
        leagues=[
            "soccer_epl", 
            "soccer_spain_la_liga", 
            "soccer_germany_bundesliga", 
            "soccer_italy_serie_a", 
            "soccer_france_ligue_one"
        ],
        save_predictions=False
    )
    
    pipeline = DailyPipeline(config=config)
    
    try:
        logger.info("Running pipeline...")
        bets = pipeline.run()
        logger.info(f"Pipeline finished. Found {len(bets)} bets.")
        
        if bets:
            for i, bet in enumerate(bets[:5]):
                print(f"{i+1}. {bet.home_team} vs {bet.away_team} - EV: {bet.ev:.2%}")
        else:
            logger.warning("No bets found! Checking intermediate steps...")
            
    except Exception as e:
        logger.error(f"Pipeline crashed: {e}", exc_info=True)

if __name__ == "__main__":
    run_debug()
