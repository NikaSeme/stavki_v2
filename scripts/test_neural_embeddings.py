
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from stavki.models.neural.multitask import MultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_neural_embeddings():
    logger.info("Generating synthetic data...")
    
    # Generate synthetic data
    n_samples = 1000
    teams = [f"Team_{i}" for i in range(20)]
    leagues = ["League_A", "League_B", "League_C"]
    
    data = pd.DataFrame({
        "HomeTeam": np.random.choice(teams, n_samples),
        "AwayTeam": np.random.choice(teams, n_samples),
        "League": np.random.choice(leagues, n_samples),
        "FTHG": np.random.randint(0, 5, n_samples),
        "FTAG": np.random.randint(0, 5, n_samples),
        "Date": pd.date_range(start="2023-01-01", periods=n_samples, freq="D"),
        # Numeric features
        "HomeEloBefore": np.random.normal(1500, 100, n_samples),
        "AwayEloBefore": np.random.normal(1500, 100, n_samples),
        "EloDiff": np.random.normal(0, 50, n_samples),
    })
    
    # Custom features list (subset of available)
    features = ["HomeEloBefore", "AwayEloBefore", "EloDiff"]
    
    logger.info("Initializing MultiTaskModel with Embeddings...")
    model = MultiTaskModel(
        hidden_dim=32,
        n_blocks=1,
        n_epochs=5,
        batch_size=32,
        features=features
    )
    
    logger.info("Training model...")
    try:
        metrics = model.fit(data, eval_ratio=0.2)
        logger.info(f"Training metrics: {metrics}")
        
        if metrics["best_eval_loss"] == float("inf"):
            logger.error("Training failed: Loss is infinite")
            return
            
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise e
    
    logger.info("Testing predictions...")
    try:
        # Create test set with some new/unseen teams
        test_data = data.head(10).copy()
        test_data.loc[0, "HomeTeam"] = "Team_New" # Unseen team
        
        preds = model.predict(test_data)
        
        assert len(preds) == len(test_data) * 3, f"Expected {len(test_data)*3} predictions, got {len(preds)}"
        
        # Check structure
        p0 = preds[0]
        logger.info(f"Sample prediction: {p0}")
        
        assert "home" in p0.probabilities
        assert "draw" in p0.probabilities
        
        logger.info("Verification Successful!")
        
    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        raise e

if __name__ == "__main__":
    test_neural_embeddings()
