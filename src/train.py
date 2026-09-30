"""
Trains and evaluates CTR prediction models:
  1. Logistic Regression baseline (with one-hot encoded categoricals)
  2. XGBoost gradient-boosted trees (native categorical support)

Metrics reported (all computed on a held-out test set, never seen during training):
  - AUC-ROC
  - Log loss
  - Precision@K (K = top-ranked impressions by predicted click probability)
  - Lift over random baseline at top-K
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

CATEGORICAL = ["ad_category", "device", "age_bucket", "placement"]
NUMERIC = ["hour_of_day", "day_of_week", "user_past_ctr", "ad_quality_score", "bid_price", "ad_position"]
TARGET = "clicked"


def precision_at_k(y_true, y_score, k):
    order = np.argsort(-y_score)
    top_k_idx = order[:k]
    return y_true.iloc[top_k_idx].mean() if hasattr(y_true, "iloc") else y_true[top_k_idx].mean()


def load_data():
    df = pd.read_csv("data/ad_impressions.csv")
    X = df[CATEGORICAL + NUMERIC]
    y = df[TARGET]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_logistic_regression(X_train, y_train):
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="passthrough")

    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_xgboost(X_train, y_train):
    X_train_enc = X_train.copy()
    for col in CATEGORICAL:
        X_train_enc[col] = X_train_enc[col].astype("category")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        enable_categorical=True,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
    )
    model.fit(X_train_enc, y_train)
    return model


def evaluate(model, X_test, y_test, name, is_xgb=False):
    X_eval = X_test.copy()
    if is_xgb:
        for col in CATEGORICAL:
            X_eval[col] = X_eval[col].astype("category")
        y_prob = model.predict_proba(X_eval)[:, 1]
    else:
        y_prob = model.predict_proba(X_eval)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)

    n = len(y_test)
    results_by_k = {}
    for k_pct in [0.01, 0.05, 0.10]:
        k = max(1, int(n * k_pct))
        p_at_k = precision_at_k(y_test, y_prob, k)
        baseline_ctr = y_test.mean()
        lift = p_at_k / baseline_ctr if baseline_ctr > 0 else float("nan")
        results_by_k[f"precision@top{int(k_pct*100)}%"] = round(float(p_at_k), 4)
        results_by_k[f"lift@top{int(k_pct*100)}%"] = round(float(lift), 2)

    metrics = {
        "model": name,
        "auc_roc": round(float(auc), 4),
        "log_loss": round(float(ll), 4),
        "baseline_ctr": round(float(y_test.mean()), 4),
        **results_by_k,
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k}: {v}")
    return metrics


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"Train CTR: {y_train.mean():.4f} | Test CTR: {y_test.mean():.4f}")

    all_metrics = []

    t0 = time.time()
    lr_model = train_logistic_regression(X_train, y_train)
    lr_train_time = time.time() - t0
    lr_metrics = evaluate(lr_model, X_test, y_test, "Logistic Regression", is_xgb=False)
    lr_metrics["train_time_sec"] = round(lr_train_time, 2)
    all_metrics.append(lr_metrics)

    t0 = time.time()
    xgb_model = train_xgboost(X_train, y_train)
    xgb_train_time = time.time() - t0
    xgb_metrics = evaluate(xgb_model, X_test, y_test, "XGBoost", is_xgb=True)
    xgb_metrics["train_time_sec"] = round(xgb_train_time, 2)
    all_metrics.append(xgb_metrics)

    # Feature importance from XGBoost
    importances = xgb_model.get_booster().get_score(importance_type="gain")
    importances_sorted = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])
    print("\n=== Top Feature Importances (XGBoost, gain) ===")
    for feat, score in importances_sorted.items():
        print(f"  {feat}: {score:.1f}")

    with open("models/metrics.json", "w") as f:
        json.dump({"results": all_metrics, "feature_importance": importances_sorted}, f, indent=2)

    import joblib
    joblib.dump(lr_model, "models/logistic_regression.pkl")
    xgb_model.save_model("models/xgboost_model.json")
    print("\nSaved models to models/ and metrics to models/metrics.json")


if __name__ == "__main__":
    main()
