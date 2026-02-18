
import sys
import logging
import os
from pathlib import Path

# Setup logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("Debug")

# Prepare paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Load Env
from dotenv import load_dotenv
load_dotenv()

try:
    from stavki.pipelines.daily import DailyPipeline, PipelineConfig
    
    logger.info("Initializing DailyPipeline...")
    # Force 0 cache age to trigger live fetch logic if cache exists
    # Or just rely on the fact that existing cache is old
    pipeline = DailyPipeline(config=PipelineConfig(min_ev=0.0))
    
    logger.info("Running Pipeline...")
    bets = pipeline.run()
    
    logger.info(f"Success! Found {len(bets)} bets.")
    
except Exception as e:
    logger.critical("CRASHED!", exc_info=True)
    sys.exit(1)
