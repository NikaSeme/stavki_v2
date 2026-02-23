import sys
sys.path.insert(0, '.')
import pandas as pd
import torch
import logging
from stavki.models.neural.multitask import MultiTaskModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_loss')

class LossLoggingMultiTaskModel(MultiTaskModel):
    def fit(self, *args, **kwargs):
        # We need to temporarily patch the forward method or loss computation
        # Since fit contains the train loop, we just copy the train loop 
        # and add our logging. But it's easier to just override fit entirely.
        
        # Let's just run it and see the initial loss scale
        pass

# Since the fit method is long, we'll patch the loss functions in the instance
df = pd.read_csv('data/features_full.csv', nrows=2500, low_memory=False)
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)

model = MultiTaskModel(n_epochs=2, learning_rate=0.001)

# We can intercept the loss instances
import gc
# Actually, the easiest way is to modify the file directly and revert it
