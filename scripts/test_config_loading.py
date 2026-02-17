
import logging
import json
from pathlib import Path
from stavki.data.loader import UnifiedDataLoader
from stavki.data.collectors.sportmonks import SportMonksClient
from stavki.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_config_loading():
    logger.info("Testing Config Extraction...")
    
    # 1. Load Expected Config
    config_path = PROJECT_ROOT / "stavki" / "config" / "leagues.json"
    with open(config_path) as f:
        expected_config = json.load(f)
    
    logger.info(f"Expected Config: {expected_config}")
    
    # 2. Test UnifiedDataLoader
    loader = UnifiedDataLoader()
    # Loader maps ID -> Name (lowercase, no underscore)
    expected_loader_map = {v: k.lower().replace("_", "") for k, v in expected_config.items()}
    
    logger.info(f"Loader Map: {loader.leagues_map}")
    
    assert loader.leagues_map == expected_loader_map, "UnifiedDataLoader leagues map mismatch"
    logger.info("UnifiedDataLoader verification successful!")
    
    # 3. Test SportMonksClient
    client = SportMonksClient(api_key="test_key")
    # Client maps Name -> ID (as in json)
    
    logger.info(f"Client League IDs: {client.league_ids}")
    
    assert client.league_ids == expected_config, "SportMonksClient league_ids mismatch"
    logger.info("SportMonksClient verification successful!")
    
    # 4. Check that hardcoded LEAGUE_IDS is gone from class
    assert not hasattr(SportMonksClient, "LEAGUE_IDS"), "SportMonksClient still has LEAGUE_IDS class attribute"
    logger.info("Hardcoded LEAGUE_IDS removal verification successful!")

if __name__ == "__main__":
    test_config_loading()
