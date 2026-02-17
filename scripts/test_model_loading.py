
import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.getcwd())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_loading():
    print("🚀 Testing Model Loading...")
    
    try:
        from stavki.pipelines.daily import DailyPipeline
        pipeline = DailyPipeline()
        
        print("Attempting to load ensemble...")
        ensemble = pipeline._load_ensemble()
        
        if ensemble:
            print(f"✅ Ensemble loaded with {len(ensemble.models)} models")
            for name in ensemble.models:
                print(f"   - {name}")
        else:
            print("❌ Ensemble failed to load (returned None)")
            
    except Exception as e:
        print(f"❌ Crash: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
