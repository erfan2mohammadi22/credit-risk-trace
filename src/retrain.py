"""
retrain.py
----------
Retrain the tuned LightGBM model with a higher boosting-round ceiling.

The previous Optuna tuning hit the 2000-round ceiling in every fold,
which means the model was still improving. This script re-runs the
5-fold CV with max_rounds=4000 and early stopping enabled, using the
best hyperparameters found by Optuna (Trial #17).
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from preprocessing import PROCESSED_PATH


# -------- Configuration --------
MODELS_DIR = Path("models")
RANDOM_STATE = 42
N_FOLDS = 5
MAX_ROUNDS = 4000
EARLY_STOP = 100

CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type",
]

# Best params from Optuna Trial #17
BEST_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "learning_rate": 0.011060468074564773,
    "num_leaves": 66,
    "max_depth": 12,
    "min_child_samples": 177,
    "feature_fraction": 0.6005837527115208,
    "bagging_fraction": 0.7481153523574066,
    "bagging_freq": 1,
    "lambda_l1": 0.6664351012585905,
    "lambda_l2": 0.001184703142174576,
    "min_gain_to_split": 0.16265484921847437,
}


# -------- Helpers --------
def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    return df


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["target", "loan_status"], errors="ignore")
    y = df["target"].values
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, y


def compute_ks(y_true, y_proba):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


# -------- Main --------
def main() -> None:
    df = load_data()
    X, y = prepare_features(df)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, kss, ginis, best_iters = [], [], [], []

    print("=" * 60)
    print(f"Retraining with max_rounds={MAX_ROUNDS}, early_stop={EARLY_STOP}")
    print("=" * 60)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature="auto")
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature="auto")

        model = lgb.train(
            BEST_PARAMS,
            dtrain,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred)
        ks = compute_ks(y_val, y_pred)
        aucs.append(auc)
        kss.append(ks)
        ginis.append(2 * auc - 1)
        best_iters.append(model.best_iteration)

        print(f"  Fold {fold}: AUC={auc:.4f}  KS={ks:.4f}  best_iter={model.best_iteration}")

    # ---- Summary ----
    print(f"\n--- Retrain Summary ---")
    print(f"AUC:  {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    print(f"KS:   {np.mean(kss):.4f}")
    print(f"Gini: {np.mean(ginis):.4f}")
    print(f"Avg best_iter: {int(np.mean(best_iters))}")

    # ---- Train deployment model on ALL data ----
    print(f"\nTraining deployment model on ALL data ...")
    final_rounds = int(np.mean(best_iters) * 1.05)
    dtrain_all = lgb.Dataset(X, y, categorical_feature="auto")
    final_model = lgb.train(BEST_PARAMS, dtrain_all, num_boost_round=final_rounds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "lightgbm_final.joblib"
    joblib.dump(final_model, path)

    print(f"Saved final model: {path}")
    print(f"Trained with {final_rounds} rounds")
    print("\nDone.")


if __name__ == "__main__":
    main()