#!/usr/bin/env python3
import sys
import os
import logging

# Add project root to path
sys.path.insert(0, os.getcwd())

from stavki.config import get_config
from stavki.interfaces.telegram_bot import create_bot

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)
    
    logger.info("Starting STAVKI Bot Service...")
    
    config = get_config()
    
    if not config.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN not set in .env")
        sys.exit(1)
        
    try:
        bot = create_bot(token=config.telegram_bot_token, min_ev=config.min_ev)
        bot.run()
    except Exception as e:
        logger.critical(f"Bot failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
