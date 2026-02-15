"""STAVKI CLI Module."""
# Re-export main from the cli.py module for entry point compatibility
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from cli import main

__all__ = ["main"]
