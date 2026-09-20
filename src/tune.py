"""
tune.py
-------
[DEPRECATED]
This script is from v1/v2 (with addr_state and issue_year).
The current pipeline uses preprocessing.py v4 which removes
those features. Kept for historical reference.

Hyperparameter tuning for LightGBM using Optuna.

Strategy:
- Subsample 25% of data for speed
- 3-Fold CV per trial
- Median pruning to stop weak trials early
- 30 trials total
- Retrain final model on full data with best params
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from preprocessing import PROCESSED_PATH


MODELS_DIR = Path("models")
RANDOM_STATE = 42
N_TRIALS = 30
TUNE_SAMPLE_FRAC = 0.25
TUNE_N_FOLDS = 3
FINAL_N_FOLDS = 5
EARLY_STOP = 50


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
    X = df.drop(columns=["target", "loan_status"], errors="ignore")
    y = df["target"].values
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, y


def compute_ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def cross_val_auc(X: pd.DataFrame, y: np.ndarray, params: dict,
                  n_folds: int, trial: optuna.Trial | None = None) -> float:
    """Return mean AUC across folds. Optionally prune weak trials."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature="auto")
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature="auto")

        model = lgb.train(
            params, dtrain, num_boost_round=1500,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)

        if trial is not None:
            trial.report(float(np.mean(aucs)), fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return float(np.mean(aucs))


def objective(trial: optuna.Trial, X: pd.DataFrame, y: np.ndarray) -> float:
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
    }
    return cross_val_auc(X, y, params, TUNE_N_FOLDS, trial=trial)


def train_final(X: pd.DataFrame, y: np.ndarray, params: dict) -> tuple:
    """Train final model with 5-fold CV and return the OOF AUC and best_iter."""
    skf = StratifiedKFold(n_splits=FINAL_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, kss, best_iters = [], [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature="auto")
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature="auto")
        model = lgb.train(
            params, dtrain, num_boost_round=2000,
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
        best_iters.append(model.best_iteration)
        print(f"  Fold {fold}: AUC={auc:.4f}  KS={ks:.4f}  best_iter={model.best_iteration}")

    print(f"\nFinal CV -> AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})  "
          f"KS: {np.mean(kss):.4f}  "
          f"Gini: {2 * np.mean(aucs) - 1:.4f}")
    return aucs, best_iters


def main() -> None:
    df = load_data()
    X, y = prepare_features(df)

    # Subsample for tuning
    print(f"Subsampling {TUNE_SAMPLE_FRAC * 100:.0f}% of data for tuning ...")
    X_tune, _, y_tune, _ = train_test_split(
        X, y,
        train_size=TUNE_SAMPLE_FRAC,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # Reset index to avoid issues with .iloc in CV
    X_tune = X_tune.reset_index(drop=True)
    print(f"Tuning sample: {X_tune.shape[0]:,} rows\n")

    # Optuna study with median pruning
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    print("=" * 60)
    print(f"Optuna tuning ({N_TRIALS} trials, {TUNE_N_FOLDS}-fold CV)")
    print("=" * 60)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        lambda t: objective(t, X_tune, y_tune),
        n_trials=N_TRIALS,
        show_progress_bar=False,
    )

    print(f"\nBest trial: #{study.best_trial.number}")
    print(f"Best AUC (tuning): {study.best_value:.4f}")
    print("\nBest params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Build final params
    best_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        **study.best_params,
    }

    # Retrain on full data with 5-fold CV to get the true performance
    print("\n" + "=" * 60)
    print(f"Final training on FULL data ({FINAL_N_FOLDS}-fold CV)")
    print("=" * 60)
    aucs, best_iters = train_final(X, y, best_params)

    # Train a single final model on ALL data for deployment
    print("\nTraining deployment model on ALL data ...")
    final_rounds = int(np.mean(best_iters) * 1.05)  # +5% rounds
    dtrain_all = lgb.Dataset(X, y, categorical_feature="auto")
    final_model = lgb.train(best_params, dtrain_all, num_boost_round=final_rounds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "lightgbm_tuned.joblib"
    joblib.dump(final_model, path)
    print(f"Saved final model: {path}")
    print(f"Trained with {final_rounds} rounds")

    print("\nDone.")


if __name__ == "__main__":
    main()