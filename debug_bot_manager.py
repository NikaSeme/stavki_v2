
import sys
import logging
import os
from pathlib import Path

# Setup logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("DebugManager")

# Prepare paths
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Load Env
from dotenv import load_dotenv
load_dotenv()

try:
    from stavki.interfaces.telegram_bot import GlobalPipelineManager
    
    logger.info("Initializing Manager...")
    manager = GlobalPipelineManager()
    
    logger.info("Calling run_scan()...")
    bets = manager.run_scan()
    
    logger.info(f"Success! Manager returned {len(bets)} bets.")
    
except Exception as e:
    logger.critical("CRASHED!", exc_info=True)
    sys.exit(1)
