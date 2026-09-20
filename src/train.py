"""
train.py
--------
Train baseline and advanced models for credit risk prediction.
Evaluates with AUC, KS, and Gini.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from preprocessing import PROCESSED_PATH


MODELS_DIR = Path("models")
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["target", "loan_status"], errors="ignore")
    y = df["target"]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical columns."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}\n")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


def compute_ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute the KS statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_proba, model_name: str) -> dict:
    auc = roc_auc_score(y_true, y_proba)
    ks = compute_ks(y_true, y_proba)
    gini = 2 * auc - 1
    print(f"\n--- {model_name} ---")
    print(f"AUC:  {auc:.4f}")
    print(f"KS:   {ks:.4f}")
    print(f"Gini: {gini:.4f}")
    return {"model": model_name, "auc": auc, "ks": ks, "gini": gini}


def train_logistic(X_train, y_train, preprocessor) -> Pipeline:
    """Train Logistic Regression inside a Pipeline."""
    print("Training Logistic Regression ...")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


class XGBWrapper:
    """
    Lightweight wrapper to combine a fitted preprocessor
    with an XGBoost classifier, exposing predict_proba.
    """

    def __init__(self, preprocessor, classifier):
        self.preprocessor = preprocessor
        self.classifier = classifier

    def predict_proba(self, X):
        Xt = self.preprocessor.transform(X)
        return self.classifier.predict_proba(Xt)

    def predict(self, X):
        Xt = self.preprocessor.transform(X)
        return self.classifier.predict(Xt)


def train_xgboost(X_train, y_train, preprocessor) -> XGBWrapper:
    """
    Train XGBoost manually (outside of sklearn Pipeline) so that
    early stopping works correctly with a validation set.

    Note: early_stopping_rounds is passed to .fit() (not the
    constructor) because XGBoost 2.0+ requires it there.
    """
    print("Training XGBoost ...")

    # Split a validation set for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    # Fit preprocessor on training data only, then transform both sets
    pre = clone(preprocessor)
    X_tr_t = pre.fit_transform(X_tr)
    X_val_t = pre.transform(X_val)

    # Train XGBoost directly
    clf = XGBClassifier(
        n_estimators=1000,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(
        X_tr_t, y_tr,
        eval_set=[(X_val_t, y_val)],
        early_stopping_rounds=30,
        verbose=False,
    )
    print(f"  Best iteration: {clf.best_iteration}")

    return XGBWrapper(pre, clf)


def save_model(model, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def main() -> None:
    df = load_data()
    X, y = split_features_target(df)

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"Train: {X_train.shape[0]:,} rows")
    print(f"Test:  {X_test.shape[0]:,} rows\n")

    preprocessor = build_preprocessor(X)

    # ---- Logistic Regression ----
    log_model = train_logistic(X_train, y_train, preprocessor)
    y_proba_log = log_model.predict_proba(X_test)[:, 1]
    results_log = evaluate(y_test, y_proba_log, "Logistic Regression")
    save_model(log_model, "logistic_regression")

    # ---- XGBoost ----
    xgb_model = train_xgboost(X_train, y_train, preprocessor)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    results_xgb = evaluate(y_test, y_proba_xgb, "XGBoost")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    summary = pd.DataFrame([results_log, results_xgb])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()