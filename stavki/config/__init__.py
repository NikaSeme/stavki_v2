"""
Configuration management for STAVKI.

Loads settings from environment variables with sensible defaults.
Configures logging with rotation to prevent unbounded log growth.
"""

import os
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project paths - self-contained, no external dependencies
# Adjusted for stavki/config/__init__.py location
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB per file
    backup_count: int = 3,
) -> None:
    """Configure logging with console output AND rotating file handler.
    
    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files (defaults to PROJECT_ROOT/logs)
        max_bytes: Max size per log file before rotation (default 5MB)
        backup_count: Number of rotated backup files to keep
    """
    log_dir = log_dir or LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers to avoid duplicates on reload
    root.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    
    # Rotating file handler (P2: prevents unbounded log growth)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "stavki.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


@dataclass
class Config:
    """Application configuration."""
    
    # API Keys
    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    sportmonks_api_key: str = field(default_factory=lambda: os.getenv("SPORTMONKS_API_KEY", ""))
    betfair_app_key: str = field(default_factory=lambda: os.getenv("BETFAIR_APP_KEY", ""))
    
    # Telegram
    telegram_bot_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_allowed_users: list[int] = field(default_factory=lambda: [
        int(x) for x in os.getenv("TELEGRAM_ALLOWED_USERS", "").split(",") if x
    ])
    
    # Database
    database_path: Path = field(default_factory=lambda: Path(
        os.getenv("DATABASE_PATH", "artifacts/stavki.db")
    ))
    
    # Mode
    dry_run: bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Kelly Staking
    kelly_fraction: float = field(default_factory=lambda: float(os.getenv("KELLY_FRACTION", "0.25")))
    max_stake_pct: float = field(default_factory=lambda: float(os.getenv("MAX_STAKE_PCT", "0.05")))
    max_daily_loss: float = field(default_factory=lambda: float(os.getenv("MAX_DAILY_LOSS", "0.20")))
    
    # EV Thresholds
    min_ev: float = 0.05
    min_edge: float = 0.02
    model_alpha: float = 0.5  # Blend weight for model vs market
    
    # Backtest / Simulation
    leagues: list = field(default_factory=list)
    market_ban_prob: float = 0.0
    
    # Retraining
    retrain_data_path: str = field(default_factory=lambda: os.getenv(
        "RETRAIN_DATA_PATH", "data/features_full.csv"
    ))
    
    def __post_init__(self):
        """Ensure directories exist and logging is configured."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        setup_logging(level=self.log_level)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global config instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Force reload configuration."""
    global _config
    load_dotenv(override=True)
    _config = Config()
    return _config
