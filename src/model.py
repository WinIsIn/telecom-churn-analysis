"""
Train and evaluate churn prediction models for KiwiTel.
Saves the best model as model.pkl.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.features import engineer_features, get_feature_columns

DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'customers.csv'
MODEL_PATH = Path(__file__).parent.parent / 'model.pkl'


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = engineer_features(df)
    features = get_feature_columns()
    X = df[features]
    y = (df['churn'] == 'Yes').astype(int)
    return X, y, features


def train():
    X, y, feature_names = load_and_prepare()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- Logistic Regression ---
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_proba = lr.predict_proba(X_test_sc)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)

    print("=" * 50)
    print("LOGISTIC REGRESSION")
    print("=" * 50)
    print(classification_report(y_test, lr_pred, target_names=['Retained', 'Churned']))
    print(f"AUC-ROC: {lr_auc:.4f}")

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)

    print("\n" + "=" * 50)
    print("RANDOM FOREST")
    print("=" * 50)
    print(classification_report(y_test, rf_pred, target_names=['Retained', 'Churned']))
    print(f"AUC-ROC: {rf_auc:.4f}")

    # --- Gradient Boosting ---
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_pred = gb.predict(X_test)
    gb_proba = gb.predict_proba(X_test)[:, 1]
    gb_auc = roc_auc_score(y_test, gb_proba)

    print("\n" + "=" * 50)
    print("GRADIENT BOOSTING")
    print("=" * 50)
    print(classification_report(y_test, gb_pred, target_names=['Retained', 'Churned']))
    print(f"AUC-ROC: {gb_auc:.4f}")

    # Pick best model
    best_model = gb if gb_auc >= rf_auc else rf
    best_auc = max(gb_auc, rf_auc)
    best_importances = pd.Series(best_model.feature_importances_, index=feature_names)

    # Feature importance — top 10
    top10 = best_importances.nlargest(10)
    print(f"\nTop 10 Feature Importances (best model: {'GradientBoosting' if gb_auc >= rf_auc else 'RandomForest'}):")
    for feat, imp in top10.items():
        print(f"  {feat:<35} {imp:.4f}")

    if best_auc <= 0.75:
        print(f"WARNING: Best AUC-ROC {best_auc:.4f} is below target 0.75")

    # Save best model with scaler metadata
    payload = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'auc_roc': best_auc,
        'lr_auc_roc': lr_auc,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\nModel saved -> {MODEL_PATH}")
    return payload


if __name__ == '__main__':
    train()
