import pandas as pd
import re

df = pd.read_csv('data/features_full.csv', nrows=1)
odds_patterns = [
    r'^B365', r'^Bb', r'^P[A-Z<>]', r'^AH',
    r'^Max', r'^Avg', r'^VC', r'^IW',
    r'^LB', r'^WH', r'^GB', r'^BS',
    r'^SB', r'^SJ', r'^BW', r'^BetH'
]
drop_pattern = re.compile('|'.join(odds_patterns))
odds_cols_to_drop = [col for col in df.columns if drop_pattern.match(col) and col not in ["PSxG"]]
df = df.drop(columns=odds_cols_to_drop)
print("PC>2.5 in dropped?", "PC>2.5" in odds_cols_to_drop)
print("PC>2.5 in df?", "PC>2.5" in df.columns)
print("Remaining cols:", len(df.columns))
print(df.columns.tolist())
