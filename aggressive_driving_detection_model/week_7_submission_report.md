#  MECHALINE AI INTERNSHIP — WEEK 7 FINAL SUBMISSION REPORT

## PROJECT INFORMATION
* **Intern Name:** Muhammad Qasim Khan
* **Team Lead:** Abdul Moiz
* **Assigned Domain:** Aggressive Driving Event Detection
* **Project:** DRIVER BEHAVIOUR & ROAD SAFETY AI
* **Submission Date:** August 17, 2026

---

## 1. Executive Summary & Objective
The objective for Week 7 was to finalize machine learning model development, cross-validation, feature importance evaluation, error analysis, and technical handover for the Aggressive Driving Events domain. Building on the Week 6 engineered dataset (`aggressive_features_cleaned.csv`), a Random Forest Classifier was trained and serialized into `aggressive_driving_model.pkl`. The model achieves 96% overall accuracy, a 0.994 ROC AUC score, and a 0.923 5-fold cross-validation F1 score.

---

## 2. Workspace Artifacts & Deliverables

 `aggressive_features_cleaned.csv` | CSV Dataset | Final dataset with engineered telematics features and target behavior labels. |
 `csv_checker.py` | Sanity script verifying schema integrity, column names, and missing values. |
 `aggressive_driving_model_trainning.py` | Training pipeline script executing data preprocessing, 80/20 train-test split, model fitting, and report generation. |
 `aggressive_driving_model.pkl` | Exported model weights and preprocessing transformers for deployment and scoring. |
 `model_checker.py` |Verification script executing 5-fold cross-validation, permutation importances, and inference testing on unseen data. |
 `week7_model_evaluation.png` | Evaluation charts including the Confusion Matrix heatmap and Feature Importance bar graph. |

---

## 3. Model Performance & Evaluation Metrics

### Model Training Pipeline Output (`aggressive_driving_model_trainning.py`)
* **Preprocessing Note:** `yaw_rate_magnitude` contained missing values and was zero-imputed.
* **Evaluation Sample Size:** 6,000 hold-out samples (4,000 Safe, 2,000 Aggressive)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Safe)** | 0.96 | 0.97 | 0.97 | 4,000 |
| **1 (Aggressive)** | 0.94 | 0.92 | 0.93 | 2,000 |
| **Accuracy** | | | **0.95** | **6,000** |
| **Macro Avg** | 0.95 | 0.94 | 0.95 | 6,000 |
| **Weighted Avg** | 0.95 | 0.95 | 0.95 | 6,000 |

* **Confusion Matrix:** `[[3882, 118], [162, 1838]]`
* **ROC AUC Score:** `0.9874`

---

### Full Model Verification Output (`model_checker.py`)
* **Evaluation Sample Size:** 30,000 validation samples (20,000 Safe, 10,000 Aggressive)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Safe)** | 0.97 | 0.98 | 0.97 | 20,000 |
| **1 (Aggressive)** | 0.95 | 0.93 | 0.94 | 10,000 |
| **Accuracy** | | | **0.96** | **30,000** |
| **Macro Avg** | 0.96 | 0.95 | 0.96 | 30,000 |
| **Weighted Avg** | 0.96 | 0.96 | 0.96 | 30,000 |

* **Confusion Matrix:** `[[19561, 439], [687, 9313]]`
* **ROC AUC Score:** `0.9943`
* **5-Fold Cross-Validation F1 Scores:** `[0.9219, 0.9163, 0.9242, 0.9253, 0.9275]`
* **Mean CV F1 Score:** **0.9230 (92.3%)**

---

### Feature Importance Ranking (Permutation Importances)
| Rank | Feature Name | Permutation Score | Domain Significance |
| :--- | :--- | :--- | :--- |
| 1 | `speed_kmph` | **0.215087** | Key velocity contextual factor scaling risk during maneuvers[cite: 1, 2]. |
| 2 | `lateral_g_force` | **0.128310** | Primary indicator for hard cornering and aggressive swerving[cite: 1, 2]. |
| 3 | `lane_drift_risk` | **0.088207** | Captures dangerous lateral sway across lane markings[cite: 1, 2]. |
| 4 | `steering_rate` | **0.007037** | Detects sudden steering wheel jerks[cite: 1, 2]. |
| 5 | `yaw_rate_magnitude` | **0.000000** | Imputed column; placeholder for gyroscope inputs[cite: 1, 2]. |

---

## 4. Driver Safety Score Mapping (0–100 Framework)
Model probabilities directly feed into the Driver Safety Score pipeline:
* **80–100 (Safe Driver):** Event probability $< 0.20$.
* **60–79 (Moderate Risk):** Event probability between $0.20 - 0.49$[cite: 1].
* **40–59 (High Risk):** Event probability between $0.50 - 0.79$[cite: 1].
* **0–39 (Very High Risk):** Event probability $\ge 0.80$[cite: 1].

---

## 5. Repository Handover Instructions
1. Run `python csv_checker.py` to verify data schemas.
2. Execute `python aggressive_driving_model_trainning.py` to retrain and update weights.
3. Execute `python model_checker.py` to run 5-fold cross validation and generate permutation importances.