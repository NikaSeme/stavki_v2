#!/usr/bin/env python3
"""
STAVKI Bot Loader
=================
Entry point to start the persistent Telegram Bot service.
Loads environment variables and launches the bot.
"""

import os
import sys
import logging
import signal
import asyncio
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.append(str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / "logs" / "bot.log")
    ]
)

logger = logging.getLogger("BotLoader")

def main():
    # Load .env
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        logger.info(f"Loaded environment from {env_path}")
    else:
        logger.warning(f".env not found at {env_path}")

    # Get Token
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.critical("TELEGRAM_BOT_TOKEN not found in environment variables!")
        sys.exit(1)

    # Import Bot Module
    try:
        from stavki.interfaces.telegram_bot import create_bot
        
        # Create and Run
        bot = create_bot(token=token)
        logger.info("Starting StavkiBot...")
        
        # The bot.run() method handles the scheduler and polling
        bot.run()
        
    except Exception as e:
        logger.critical(f"Failed to start bot: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure logs dir exists
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    main()
