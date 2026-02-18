
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from stavki.models.neural.multitask import MultiTaskModel
from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
from stavki.models.poisson.dixon_coles import DixonColesModel
from stavki.models.catboost.catboost_model import CatBoostModel

def main():
    print("🔬 Diagnostic: Ensemble Model Analysis")
    print("=======================================")
    
    # 1. Load Data
    data_path = PROJECT_ROOT / "data" / "features_full.csv"
    df = pd.read_csv(data_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Downcast
    fcols = df.select_dtypes('float').columns
    df[fcols] = df[fcols].astype(np.float32)
    
    # Split
    train_ratio = 0.60
    val_ratio = 0.20
    train_end = int(len(df) * train_ratio)
    val_end = int(len(df) * (train_ratio + val_ratio))
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    
    print(f"Validation Set: {len(val_df)} matches")
    
    y_true = np.where(
        val_df["FTHG"] > val_df["FTAG"], 0,
        np.where(val_df["FTHG"] < val_df["FTAG"], 2, 1)
    )
    
    # 2. Load Models
    models_dir = PROJECT_ROOT / "models"
    from stavki.models.baseline.market_implied import MarketImpliedModel

    model_files = {
        "neural": "neural_multitask",
        "lightgbm": "LightGBM_1X2.pkl",
        "poisson": "dixon_coles.pkl",
        "market": "baseline", # Special case
    }
    
    results = []
    start_probs = {} # Store Home Win probs for correlation
    
    from stavki.models.base import Market
    
    for name, filename in model_files.items():
        print(f"\nTesting {name}...")
        path = models_dir / filename
        
        try:
            if name == "neural":
                 model = MultiTaskModel.load(path)
            elif name == "lightgbm":
                 model = LightGBMModel.load(path)
            elif name == "poisson":
                 model = DixonColesModel.load(path)
            elif name == "market":
                 model = MarketImpliedModel()
            else:
                continue
                
            if name == "poisson" and hasattr(model, "predict_rolling"):
                print("    (Using Dynamic Rolling Prediction)")
                preds = model.predict_rolling(val_df, update=True)
            else:
                preds = model.predict(val_df)

            p_1x2 = [p for p in preds if p.market == Market.MATCH_WINNER]
            
            # Align
            # Assuming aligned list (dangerous but standard for our Base)
            probs = np.array([
                [p.probabilities["home"], p.probabilities["draw"], p.probabilities["away"]]
                for p in p_1x2
            ])
            
            if len(probs) != len(val_df):
                print(f"  Length mismatch: {len(probs)} vs {len(val_df)}")
                continue

            loss = log_loss(y_true, probs)
            acc = accuracy_score(y_true, np.argmax(probs, axis=1))
            
            print(f"  Log Loss: {loss:.4f}")
            print(f"  Accuracy: {acc:.4%}")
            
            results.append({"name": name, "loss": loss})
            start_probs[name] = probs[:, 0] # Home win prob
            
        except Exception as e:
            print(f"  Failed: {e}")

    # 3. Correlation Matrix
    if len(start_probs) > 1:
        print("\n🔗 Correlation Matrix (Home Probabilities):")
        df_corr = pd.DataFrame(start_probs)
        print(df_corr.corr())
        
        # 4. Error Correlation (Residuals)
        # Residual = Prob(TrueClass) - 1.0
        # If models make same errors, residuals are correlated
        print("\n📉 Residual Correlation (Errors):")
        residuals = {}
        for name, probs_home in start_probs.items():
            # Need prob of TRUE class
            # This is hard to reconstruct from just home prob
            # Let's approximate with Home Prob error (if Home won, prob-1, else prob-0)
            # True Home Win = (y_true == 0)
            is_home_win = (y_true == 0).astype(int)
            res = probs_home - is_home_win
            residuals[name] = res
            
        print(pd.DataFrame(residuals).corr())

if __name__ == "__main__":
    main()
