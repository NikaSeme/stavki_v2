#!/usr/bin/env python3
import sys
import os
import logging
from pathlib import Path

# Add project root to path robustly
# Current: scripts/run_bot.py -> Root: ..
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import from stavki
try:
    from stavki.config import get_config
    from stavki.interfaces.telegram_bot import create_bot
except ImportError as e:
    print(f"CRITICAL: Failed to import stavki modules: {e}")
    sys.exit(1)

def main():
    # Setup logging
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "bot.log")
        ]
    )
    logger = logging.getLogger("stavki_bot")
    
    logger.info("--------------------------------")
    logger.info("Starting STAVKI Bot Service...")
    logger.info(f"Project Root: {PROJECT_ROOT}")
    
    try:
        config = get_config()
        
        if not config.telegram_bot_token:
            logger.error("TELEGRAM_BOT_TOKEN not set in .env")
            print("Error: TELEGRAM_BOT_TOKEN missing")
            sys.exit(1)
            
        bot = create_bot(token=config.telegram_bot_token, min_ev=config.min_ev)
        
        logger.info("Bot created, starting polling...")
        bot.run()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.critical(f"Bot failed with unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
