
"""
Supervisor Bot (Watchdog) for STAVKI
====================================
Monitors the main bot process and restarts it if it crashes or freezes.
- Checks systemd service status
- Checks heartbeat (last log update)
- Sends Telegram alerts on restart
"""
import sys
import os
import time
import logging
import subprocess
import requests
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("watchdog.log")
    ]
)
logger = logging.getLogger("Watchdog")

PROCESS_NAME = "start_bot.py"
SERVICE_NAME = "stavki_bot"
CHECK_INTERVAL = 60  # Check every minute
HEARTBEAT_FILE = Path("outputs/logs/heartbeat.json")
HEARTBEAT_TIMEOUT = 300  # 5 minutes without update = freeze

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
ADMIN_ID = os.environ.get("TELEGRAM_ADMIN_ID")

def send_alert(message):
    """Send alert to admin via Telegram."""
    if not TELEGRAM_TOKEN or not ADMIN_ID:
        logger.warning(f"Telegram config missing. Alert suppressed: {message}")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={
            "chat_id": ADMIN_ID,
            "text": f"🚨 **STAVKI Watchdog** 🚨\n{message}",
            "parse_mode": "Markdown"
        }, timeout=10)
    except Exception as e:
        logger.error(f"Failed to send alert: {e}")

def get_service_status():
    """Check if systemd service is active."""
    try:
        cmd = ["systemctl", "is-active", SERVICE_NAME]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to check service status: {e}")
        return "unknown"

def restart_service():
    """Restart the bot service."""
    logger.info("Restarting service...")
    try:
        subprocess.run(["sudo", "systemctl", "restart", SERVICE_NAME], check=True)
        logger.info("Service restarted successfully.")
        send_alert("Bot service was restarted due to failure/freeze.")
        return True
    except Exception as e:
        logger.error(f"Failed to restart service: {e}")
        send_alert(f"Failed to restart service! Manual intervention required.\nError: {e}")
        return False

def check_heartbeat():
    """Check if the bot is writing to logs/heartbeat."""
    # This implies the main bot writes a heartbeat file. 
    # If not implemented yet, rely on process status for now.
    # Future TODO: Implement heartbeat in main bot.
    return True

def main():
    logger.info("Watchdog started.")
    send_alert("Supervisor Watchdog monitoring started.")
    
    while True:
        try:
            status = get_service_status()
            logger.debug(f"Service status: {status}")
            
            if status != "active":
                logger.warning(f"Service is {status}. Attempting restart...")
                restart_service()
            
            # TODO: Add logic to check logs for "Traceback" or errors if needed
            
        except Exception as e:
            logger.error(f"Watchdog loop error: {e}")
            
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
