
from pathlib import Path
from stavki.models.base import BaseModel

def inspect():
    models_dir = Path("models")
    for p in models_dir.glob("*.pkl"):
        print(f"\n--- {p.name} ---")
        try:
            model = BaseModel.load(p)
            print(f"Loaded {model.name}")
            if hasattr(model, "features"):
                feats = model.features
                print(f"Feature count: {len(feats)}")
                if len(feats) < 60:
                    print(f"Features: {feats}")
                if len(feats) != len(set(feats)):
                    print("DUPLICATES FOUND in features!")
                    seen = set()
                    dupes = {x for x in feats if x in seen or seen.add(x)}
                    print(f"Duplicates: {dupes}")
                else:
                    print("No duplicates in features.")
            else:
                print("No 'features' attribute.")
                
            if hasattr(model, "cat_features"):
                 print(f"Cat features count: {len(model.cat_features)}")
                 print(f"Cat features: {model.cat_features}")
                 
        except Exception as e:
            print(f"Failed to load: {e}")

if __name__ == "__main__":
    inspect()
