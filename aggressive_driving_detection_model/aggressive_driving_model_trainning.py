import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.inspection import permutation_importance
import joblib

sns.set_theme(style='whitegrid')

# 1. Load the finalized Week 6 dataset (use cleaned CSV in this folder)
DATA_PATH = "aggressive_features_cleaned.csv"
if not os.path.exists(DATA_PATH):
    print(f"ERROR: {DATA_PATH} not found in current folder: {os.getcwd()}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
# normalize column names
df.columns = df.columns.str.strip()

# steering_rate: absolute delta of steering_angle
if 'steering_rate' not in df.columns and 'steering_angle' in df.columns:
    df['steering_rate'] = df['steering_angle'].diff().abs().fillna(0)

# lateral_g_force: use lateral accel (accel_y) as lateral g-force proxy
if 'lateral_g_force' not in df.columns and 'accel_y' in df.columns:
    df['lateral_g_force'] = df['accel_y'].abs()

# yaw_rate_magnitude: absolute yaw rate from gyro_z
if 'yaw_rate_magnitude' not in df.columns and 'gyro_z' in df.columns:
    df['yaw_rate_magnitude'] = df['gyro_z'].abs()

# lane_drift_risk: absolute lane deviation as a simple risk score
if 'lane_drift_risk' not in df.columns and 'lane_deviation' in df.columns:
    df['lane_drift_risk'] = df['lane_deviation'].abs()

# optional: save derived CSV so training uses already-computed features
df.to_csv('aggressive_features_cleaned.csv', index=False)

# 2. Define features (engineered metrics) and preferred target label
feature_cols = ['steering_rate', 'lateral_g_force', 'yaw_rate_magnitude', 'lane_drift_risk', 'speed_kmph']
# Try to auto-detect the target column if not present
preferred_targets = ['behavior_label', 'is_aggressive', 'label']
target_col = None
for t in preferred_targets:
    if t in df.columns:
        target_col = t
        break

if target_col is None:
    print("ERROR: No target column found. Expected one of:", preferred_targets)
    print("Columns available:", list(df.columns))
    sys.exit(1)

# If the dataset uses a textual 'label' column, convert to binary 'is_aggressive'
if target_col == 'label':
    y = df['label'].astype(str).str.contains('aggressive', case=False, na=False).astype(int)
    print("Derived binary target from 'label' column (0=safe,1=aggressive)")
else:
    y = df[target_col]

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Attempt to derive engineered features when base columns exist
if 'steering_rate' not in df.columns and 'steering_angle' in df.columns:
    df['steering_rate'] = df['steering_angle'].diff().abs().fillna(0)

if 'lateral_g_force' not in df.columns and 'accel_y' in df.columns:
    df['lateral_g_force'] = df['accel_y'].abs()

if 'yaw_rate_magnitude' not in df.columns and 'gyro_z' in df.columns:
    df['yaw_rate_magnitude'] = df['gyro_z'].abs()

if 'lane_drift_risk' not in df.columns and 'lane_deviation' in df.columns:
    df['lane_drift_risk'] = df['lane_deviation'].abs()

# Validate feature columns exist (after derivation)
missing_feats = [f for f in feature_cols if f not in df.columns]
if missing_feats:
    print(f"Warning: Some engineered feature columns are still missing: {missing_feats}")
    print("Available columns:", list(df.columns))
    # We'll continue and attempt to fill missing features with zeros to allow training

X = df.reindex(columns=feature_cols).copy()

# 3. Handle missing values: RandomForest in scikit-learn does not accept NaNs
# If any feature column has no observed values (all NaN), fill with zeros and warn
all_na = [c for c in feature_cols if not X[c].notna().any()]
if all_na:
    print(f"Warning: features with all missing values: {all_na}. Filling with 0s.")
    X[all_na] = X[all_na].fillna(0)

imputer = SimpleImputer(strategy='median')
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# Ensure y has no missing values and align
mask = ~pd.isna(y)
X_imp = X_imp.loc[mask].reset_index(drop=True)
y = pd.Series(y.loc[mask]).reset_index(drop=True)

if y.nunique() < 2:
    print("ERROR: Target has fewer than 2 classes after preprocessing. Cannot train.")
    sys.exit(1)

# 4. Train-Test Split (80% Training, 20% Unseen Testing) - stratify when possible
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    # fallback to non-stratified split if stratify fails (e.g., tiny class counts)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imp, y, test_size=0.2, random_state=42, stratify=None
    )

# 5. Train open-source Random Forest Model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 6. Predictions on Unseen Test Data
y_pred = model.predict(X_test)
y_prob = None
if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
    y_prob = model.predict_proba(X_test)[:, 1]

# 7. Print Performance Metrics
print("=== WEEK 7 MODEL EVALUATION REPORT ===")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix:\n', cm)
if y_prob is not None:
    try:
        print('ROC AUC:', roc_auc_score(y_test, y_prob))
    except Exception:
        pass

# 8. Generate Visualizations (Confusion Matrix & Feature Importance)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Safe', 'Aggressive'], yticklabels=['Safe', 'Aggressive'])
axes[0].set_title('Model Error Analysis (Confusion Matrix)')
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('Actual Class')

# Feature Importance Bar Plot
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
importances.plot(kind='barh', ax=axes[1], color='teal')
axes[1].set_title('Top Aggressive Driving Indicators')
axes[1].set_xlabel('Importance Score')

plt.tight_layout()
out_img = 'week7_model_evaluation.png'
plt.savefig(out_img, dpi=150)
plt.show()

# 9. Export Trained Model + preprocessor for Handover
export_obj = {'model': model, 'imputer': imputer, 'features': feature_cols, 'target_col': target_col}
out_model = 'aggressive_driving_model.pkl'
joblib.dump(export_obj, out_model)
print(f"✅ Model and preprocessing saved to '{out_model}'")