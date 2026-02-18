import json
from pathlib import Path

def main():
    root = Path(".")
    models_dir = root / "models"
    weights_path = root / "stavki" / "config" / "leagues.json" # The one output by retrain_system
    
    if not weights_path.exists():
        print(f"Error: {weights_path} not found")
        return

    with open(weights_path) as f:
        data = json.load(f)

    # Map keys from retrain_system style to EnsemblePredictor style
    # Based on constants in predictor.py and model definitions
    key_map = {
        "poisson": "DixonColes",
        "lightgbm": "LightGBM_1X2",
        "catboost": "CatBoost_1X2",
        "neural": "NeuralMultiTask",
        "goals_regressor": "GoalsRegressor"
    }

    new_data = {}
    
    # Handle both structures (legacy vs new)
    # New: {"epl": {"1x2": ...}} or just {"epl": {"poisson": 0.5}}?
    # retrain_system output was: {'poisson': 0.0, ...} per league?
    # Let's inspect the structure in the file by printing it first if needed, 
    # but based on logs it seemed to be a dict of leagues?
    # Log: "Updated epl: {'poisson': 0.0...}"
    # So top level keys are league names?
    
    # Let's assume top level is Leagues (or "global" + "per_league")
    # retrain_system.py seems to save `league_weights` directly.
    
    for league, market_weights in data.items():
        if isinstance(market_weights, dict):
            new_market_weights = {}
            for market, weights in market_weights.items():
                if isinstance(weights, dict):
                    # Market level -> Model weights
                    new_weights = {}
                    for model_key, val in weights.items():
                        new_key = key_map.get(model_key, model_key)
                        new_weights[new_key] = val
                    new_market_weights[market] = new_weights
                else:
                    # Maybe flat structure? Copy as is
                    new_market_weights[market] = weights
            new_data[league] = new_market_weights
        else:
            new_data[league] = market_weights
            
    # Save to league_weights.json (standard name)
    out_path = models_dir / "league_weights.json"
    with open(out_path, "w") as f:
        json.dump(new_data, f, indent=2)
        
    print(f"Successfully mapped weights and saved to {out_path}")
    print("New keys:", list(new_data.values())[0].keys() if new_data else "Empty")

if __name__ == "__main__":
    main()
