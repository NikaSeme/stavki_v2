"""
User Settings Management
========================

Handles persistence of user-specific configuration (EV threshold, bankroll).
Stores data in a local JSON file.
Thread-safe and atomic to prevent corruption.
Includes automatic backup for corrupted files.
"""

import json
import logging
import os
import shutil
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
        self._load_failed = False
        
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
                    self.settings = {
                        str(k): UserConfig.from_dict(v) 
                        for k, v in data["users"].items()
                    }
                    self.global_config = {k: v for k, v in data.items() if k != "users"}
                else:
                    # Legacy or flat structure check
                    self.settings = {
                        str(k): UserConfig.from_dict(v) 
                        for k, v in data.items() 
                        if isinstance(v, dict)
                    }
                    self.global_config = {}

                logger.info(f"Loaded settings for {len(self.settings)} users.")
                self._load_failed = False
                
            except Exception as e:
                logger.error(f"Failed to load user settings: {e}")
                # Backup corrupted file
                backup_path = self.storage_path.with_suffix(".corrupt")
                try:
                    shutil.copy(self.storage_path, backup_path)
                    logger.warning(f"Backed up corrupted settings to {backup_path}")
                except Exception as ex:
                    logger.error(f"Failed to backup corrupted settings: {ex}")
                
                # Start fresh but flag it? 
                # Actually if we backed up, we can safely overwrite on next save.
                self.settings = {}
                self._load_failed = True # Just for info

    def _save(self):
        """Save settings to JSON file atomically."""
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
                    f.flush()
                    os.fsync(f.fileno()) # Force write to disk
                
                os.replace(temp_path, self.storage_path)
                logger.info("User settings saved.")
            except Exception as e:
                logger.error(f"Failed to save user settings: {e}")

    def get_settings(self, chat_id: int) -> UserConfig:
        """Get settings for a user, returning defaults if not found."""
        with self._lock:
            chat_str = str(chat_id)
            if chat_str not in self.settings:
                # Implicitly create default config? 
                # Let's not save yet, only when they modify.
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
