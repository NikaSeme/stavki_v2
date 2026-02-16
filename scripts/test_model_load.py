
import sys
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

def test_load():
    models_dir = Path("models")
    pkl_files = list(models_dir.glob("*.pkl"))
    
    print(f"Found {len(pkl_files)} model files: {[p.name for p in pkl_files]}")
    
    for p in pkl_files:
        print(f"\n--- Loading {p.name} ---")
        try:
            from stavki.models.base import BaseModel
            obj = BaseModel.load(p)
            
            print(f"SUCCESS: Loaded {type(obj)}")
            if hasattr(obj, "name"):
                print(f"Model Name: {obj.name}")
            if hasattr(obj, "is_fitted"):
                print(f"Fitted: {obj.is_fitted}")
        except Exception as e:
            print(f"FAILURE: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_load()
