"""
User Settings Management
========================

Handles persistence of user-specific configuration (EV threshold, bankroll).
Stores data in a local JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_MIN_EV = 0.05
DEFAULT_BANKROLL = 1000.0


@dataclass
class UserConfig:
    """Configuration for a specific user/chat."""
    min_ev: float = DEFAULT_MIN_EV
    bankroll: float = DEFAULT_BANKROLL
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserConfig':
        return cls(
            min_ev=data.get("min_ev", DEFAULT_MIN_EV),
            bankroll=data.get("bankroll", DEFAULT_BANKROLL),
        )


class UserSettingsManager:
    """Manages loading and saving of user settings."""
    
    def __init__(self, storage_path: str = "config/user_settings.json"):
        # Resolve path relative to project root if needed, or absolute
        # Resolve path relative to project root
        # This file is in stavki/config/user_settings.py
        # Root is 3 levels up: ../../..
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.storage_path = root_dir / storage_path if not Path(storage_path).is_absolute() else Path(storage_path)
        self.settings: Dict[str, UserConfig] = {}
        self._load()

    def _load(self):
        """Load settings from JSON file."""
        if not self.storage_path.exists():
            logger.info(f"No user settings found at {self.storage_path}, using defaults.")
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
                for chat_id, config_data in data.items():
                    self.settings[str(chat_id)] = UserConfig.from_dict(config_data)
            logger.info(f"Loaded settings for {len(self.settings)} users.")
        except Exception as e:
            logger.error(f"Failed to load user settings: {e}")
            # Backup corrupt file?
            # For now, just log error and start with empty/defaults to avoid crash

    def _save(self):
        """Save settings to JSON file."""
        try:
            # Ensure directory exists
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {k: v.to_dict() for k, v in self.settings.items()}
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info("User settings saved.")
        except Exception as e:
            logger.error(f"Failed to save user settings: {e}")

    def get_settings(self, chat_id: int) -> UserConfig:
        """Get settings for a user, returning defaults if not found."""
        return self.settings.get(str(chat_id), UserConfig())

    def update_ev(self, chat_id: int, min_ev: float):
        """Update EV threshold for a user."""
        chat_str = str(chat_id)
        config = self.settings.get(chat_str, UserConfig())
        config.min_ev = min_ev
        self.settings[chat_str] = config
        self._save()
        logger.info(f"Updated min_ev for {chat_id} to {min_ev}")

    def update_bankroll(self, chat_id: int, bankroll: float):
        """Update bankroll for a user."""
        chat_str = str(chat_id)
        config = self.settings.get(chat_str, UserConfig())
        config.bankroll = bankroll
        self.settings[chat_str] = config
        self._save()
        logger.info(f"Updated bankroll for {chat_id} to {bankroll}")
