"""
train_advanced.py
-----------------
[DEPRECATED]
This script is from v1/v2 (with addr_state and issue_year).
The current pipeline uses preprocessing.py v4 which removes
those features. Kept for historical reference.

Advanced training with LightGBM, native categorical features,
and Stratified 5-Fold Cross-Validation.
"""atified 5-Fold Cross-Validation.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from preprocessing import PROCESSED_PATH


MODELS_DIR = Path("models")
RANDOM_STATE = 42
N_FOLDS = 5


# Columns to treat as categorical (native LightGBM support)
CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "addr_state", "application_type",
]


def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    return df


def prepare_features(df: pd.DataFrame):
    """Prepare X, y and cast categorical columns to category dtype."""
    X = df.drop(columns=["target", "loan_status"], errors="ignore")
    y = df["target"].values

    # Cast categorical columns to pandas category dtype
    existing_cat = [c for c in CATEGORICAL_COLS if c in X.columns]
    for col in existing_cat:
        X[col] = X[col].astype("category")

    print(f"Features: {X.shape[1]}")
    print(f"Categorical (native): {len(existing_cat)} -> {existing_cat}\n")
    return X, y


def compute_ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def make_lgb_params(overrides: dict | None = None) -> dict:
    """Base LightGBM parameters."""
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "min_gain_to_split": 0.01,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    if overrides:
        params.update(overrides)
    return params


def cross_validate(X: pd.DataFrame, y: np.ndarray, params: dict) -> dict:
    """Stratified K-Fold CV. Returns mean AUC, KS, Gini and OOF predictions."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(X))
    aucs, kss, ginis = [], [], []
    best_iters = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature="auto")
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature="auto")

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_proba = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = y_proba

        auc = roc_auc_score(y_val, y_proba)
        ks = compute_ks(y_val, y_proba)
        gini = 2 * auc - 1
        aucs.append(auc)
        kss.append(ks)
        ginis.append(gini)
        best_iters.append(model.best_iteration)
        print(f"  Fold {fold}: AUC={auc:.4f}  KS={ks:.4f}  best_iter={model.best_iteration}")

    results = {
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "ks_mean": float(np.mean(kss)),
        "ks_std": float(np.std(kss)),
        "gini_mean": float(np.mean(ginis)),
        "best_iter_mean": int(np.mean(best_iters)),
        "oof_preds": oof_preds,
    }
    return results


def train_final(X: pd.DataFrame, y: np.ndarray, params: dict, num_rounds: int):
    """Train final model on all data with the average best iteration."""
    print(f"\nTraining final model on all data ({num_rounds} rounds) ...")
    dtrain = lgb.Dataset(X, y, categorical_feature="auto")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
    )
    return model


def save_model(model, name: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved model: {path}")


def main() -> None:
    df = load_data()
    X, y = prepare_features(df)

    params = make_lgb_params()

    print("=" * 60)
    print("Stratified 5-Fold Cross-Validation")
    print("=" * 60)
    cv = cross_validate(X, y, params)

    print("\n--- CV Summary ---")
    print(f"AUC:  {cv['auc_mean']:.4f} +/- {cv['auc_std']:.4f}")
    print(f"KS:   {cv['ks_mean']:.4f} +/- {cv['ks_std']:.4f}")
    print(f"Gini: {cv['gini_mean']:.4f}")

    final = train_final(X, y, params, num_rounds=cv["best_iter_mean"])
    save_model(final, "lightgbm_cv")

    print("\nDone.")


if __name__ == "__main__":
    main()