"""
User Settings Management
========================

Handles persistence of user-specific configuration (EV threshold, bankroll).
Stores data in a local JSON file.
Thread-safe and atomic to prevent corruption.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional
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
        # Safely handle missing keys or wrong types
        try:
            return cls(
                min_ev=float(data.get("min_ev", DEFAULT_MIN_EV)),
                bankroll=float(data.get("bankroll", DEFAULT_BANKROLL)),
            )
        except (ValueError, TypeError):
             return cls()


class UserSettingsManager:
    """Manages loading and saving of user settings."""
    
    def __init__(self, storage_path: str = "config/user_settings.json"):
        # Resolve path relative to project root
        root_dir = Path(__file__).resolve().parent.parent.parent
        self.storage_path = root_dir / storage_path if not Path(storage_path).is_absolute() else Path(storage_path)
        
        self.settings: Dict[str, UserConfig] = {}
        self.global_config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        self._load()

    def _load(self):
        """Load settings from JSON file."""
        with self._lock:
            if not self.storage_path.exists():
                logger.info(f"No user settings found at {self.storage_path}, starting fresh.")
                return
            
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    
                # Handle "users" key structure
                if "users" in data:
                    users_data = data["users"]
                    self.global_config = {k: v for k, v in data.items() if k != "users"}
                else:
                    # Legacy or flat structure check - if keys look like chat IDs (integers)
                    # For safety, if it's not structured, assume it's flat users
                    users_data = data
                    self.global_config = {}

                for chat_id, config_data in users_data.items():
                    if isinstance(config_data, dict):
                         self.settings[str(chat_id)] = UserConfig.from_dict(config_data)
                         
                logger.info(f"Loaded settings for {len(self.settings)} users.")
            except Exception as e:
                logger.error(f"Failed to load user settings: {e}")
                # Don't crash, just start with empty settings if file is garbage
                self.settings = {}

    def _save(self):
        """Save settings to JSON file slightly atomically."""
        with self._lock:
            try:
                # Ensure directory exists
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                
                output_data = {
                    "users": {k: v.to_dict() for k, v in self.settings.items()},
                    **self.global_config
                }
                
                # Write to temp file then rename
                temp_path = self.storage_path.with_suffix(".tmp")
                with open(temp_path, "w") as f:
                    json.dump(output_data, f, indent=2)
                
                os.replace(temp_path, self.storage_path)
                logger.info("User settings saved.")
            except Exception as e:
                logger.error(f"Failed to save user settings: {e}")

    def get_settings(self, chat_id: int) -> UserConfig:
        """Get settings for a user, returning defaults if not found."""
        with self._lock:
            chat_str = str(chat_id)
            if chat_str not in self.settings:
                # Create default entry? No, just return default object.
                # Only create when they customize.
                return UserConfig()
            return self.settings[chat_str]

    def update_ev(self, chat_id: int, min_ev: float):
        """Update EV threshold for a user."""
        with self._lock:
            chat_str = str(chat_id)
            config = self.settings.get(chat_str, UserConfig())
            config.min_ev = min_ev
            self.settings[chat_str] = config
            self._save()
            logger.info(f"Updated min_ev for {chat_id} to {min_ev}")

    def update_bankroll(self, chat_id: int, bankroll: float):
        """Update bankroll for a user."""
        with self._lock:
            chat_str = str(chat_id)
            config = self.settings.get(chat_str, UserConfig())
            config.bankroll = bankroll
            self.settings[chat_str] = config
            self._save()
            logger.info(f"Updated bankroll for {chat_id} to {bankroll}")
