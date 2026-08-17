import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

# load artifacts
obj = joblib.load('aggressive_driving_model.pkl')
model = obj['model']
imputer = obj['imputer']
features = obj['features']

# load raw/cleaned csv and derive same features as training script
df = pd.read_csv('aggressive_features_cleaned.csv')
df.columns = df.columns.str.strip()
# derive steering_rate, lateral_g_force, yaw_rate_magnitude, lane_drift_risk if needed
if 'steering_rate' not in df and 'steering_angle' in df:
    df['steering_rate'] = df['steering_angle'].diff().abs().fillna(0)
if 'lateral_g_force' not in df and 'accel_y' in df:
    df['lateral_g_force'] = df['accel_y'].abs()
if 'yaw_rate_magnitude' not in df and 'gyro_z' in df:
    df['yaw_rate_magnitude'] = df['gyro_z'].abs()
if 'lane_drift_risk' not in df and 'lane_deviation' in df:
    df['lane_drift_risk'] = df['lane_deviation'].abs()

# build X,y same as training
if 'is_aggressive' in df.columns:
    y = df['is_aggressive']
elif 'behavior_label' in df.columns:
    y = df['behavior_label']
else:
    y = df['label'].astype(str).str.contains('aggressive', case=False, na=False).astype(int)

X = df.reindex(columns=features).fillna(0)
X_imp = pd.DataFrame(imputer.transform(X), columns=features)

# predict and report
y_pred = model.predict(X_imp)
print(classification_report(y, y_pred))
print('Confusion matrix:\\n', confusion_matrix(y, y_pred))
if hasattr(model, 'predict_proba') and len(y.unique()) == 2:
    probs = model.predict_proba(X_imp)[:, 1]
    print('ROC AUC:', roc_auc_score(y, probs))

# cross-validated estimate (stratified)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_imp, y, cv=cv, scoring='f1')
print('5-fold F1 scores:', scores, 'mean:', scores.mean())

# permutation importance (robust)
perm = permutation_importance(model, X_imp, y, n_repeats=10, random_state=42, n_jobs=2)
imp_df = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
print('Permutation importances:\\n', imp_df)