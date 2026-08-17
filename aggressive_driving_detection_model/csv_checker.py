import pandas as pd
from difflib import get_close_matches
df = pd.read_csv('aggressive_features_cleaned.csv')
print('Columns:', df.columns.tolist())
print(df.head(3).to_string())
expected = ['steering_rate','lateral_g_force','yaw_rate_magnitude','lane_drift_risk','speed_kmph']
for e in expected:
    if e not in df.columns:
        print(f"Missing: {e}  -> suggestions:", get_close_matches(e, df.columns, n=5, cutoff=0.5))
# target candidates
for t in ['behavior_label','is_aggressive','label']:
    if t in df.columns:
        print('Found target column:', t)
        break
else:
    print('No standard target found; candidate columns containing label/behavior/aggress:',
          [c for c in df.columns if any(k in c.lower() for k in ['label','behavior','aggress'])])
