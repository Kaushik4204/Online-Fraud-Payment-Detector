# fraud_train.py (Improved version based on recommendations)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import joblib
import matplotlib.pyplot as plt

# === 📥 Load Dataset ===
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

# === 🧹 Preprocessing ===
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)
df['type'] = df['type'].astype('category').cat.codes

# === 🛠 Feature Engineering ===
df['balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['transfer_ratio'] = df.apply(lambda row: row['amount'] / row['oldbalanceOrg'] if row['oldbalanceOrg'] > 0 else 0, axis=1)
df['net_transfer'] = df['newbalanceDest'] - df['oldbalanceDest']  # Additional useful feature

X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# === 🔄 Scale Features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# === 📊 Handle Class Imbalance with SMOTE ===
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# === 🔀 Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42
)

# === 🔥 LightGBM Classifier ===
model = LGBMClassifier(
    n_estimators=300,
    class_weight='balanced',
    random_state=42,
    learning_rate=0.05,
    max_depth=12,
    num_leaves=31,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1
)

model.fit(X_train, y_train)

# === 📈 Evaluate Model with Threshold Tuning ===
y_proba = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

# Find best threshold by maximizing F1 score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Best Threshold (F1 Max): {optimal_threshold:.4f}")

y_pred = (y_proba >= optimal_threshold).astype(int)

# === 📋 Evaluation Report ===
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === 💾 Save Model ===
joblib.dump(model, 'fraud_model.pkl')
