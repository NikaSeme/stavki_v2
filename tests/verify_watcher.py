import sys
from pathlib import Path
import logging

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_watcher")

from stavki.pipelines.daily import DailyPipeline

def main():
    logger.info("Initializing DailyPipeline...")
    pipeline = DailyPipeline()
    
    logger.info("Loading Ensemble...")
    ensemble = pipeline._load_ensemble()
    
    if not ensemble:
        logger.error("Failed to load ensemble!")
        return
        
    logger.info(f"Main Models: {list(ensemble.models.keys())}")
    logger.info(f"Shadow Models: {list(ensemble.shadow_models.keys())}")
    
    if "V3_Watcher" in ensemble.shadow_models:
        logger.info("✅ V3_Watcher is present in shadow models as expected.")
        v3 = ensemble.shadow_models["V3_Watcher"]
        logger.info(f"   V3 Fitted Status: {v3.is_fitted}")
        if not v3.is_fitted:
            logger.warning("   ⚠️ V3 Model is NOT FITTED (Expected if no weights file found)")
    else:
        logger.error("❌ V3_Watcher NOT found in shadow models!")

if __name__ == "__main__":
    main()
