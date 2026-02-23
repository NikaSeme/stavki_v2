from stavki.models.gradient_boost.lightgbm_model import LightGBMModel
import pandas as pd
import re
import pickle

df = pd.read_csv('data/features_full.csv', nrows=100)
df["target"] = 0

odds_patterns = [
    r'^B365', r'^Bb', r'^P[A-Z<>]', r'^AH',
    r'^Max', r'^Avg', r'^VC', r'^IW',
    r'^LB', r'^WH', r'^GB', r'^BS',
    r'^SB', r'^SJ', r'^BW', r'^BetH'
]
drop_pattern = re.compile('|'.join(odds_patterns))
odds_cols_to_drop = [col for col in df.columns if drop_pattern.match(col) and col not in ["PSxG"]]
df = df.drop(columns=odds_cols_to_drop)

lgb = LightGBMModel()
lgb.fit(df, eval_ratio=0.1)

lgb.save("test_lgb.pkl")
state2 = pickle.load(open("test_lgb.pkl", "rb"))
print("File metadata has PC>2.5?", "PC>2.5" in state2["metadata"]["features"])
print("Actual features saved:", len(state2["metadata"]["features"]))
