"""
ablation.py
-----------
Ablation Study: separate the effects of learning_rate and max_rounds.

Experiments:
- v5: learning_rate=0.011, max_rounds=8000
- v6: learning_rate=0.025, max_rounds=8000

Goal: Identify the primary bottleneck that prevented v4 from converging.
"""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from preprocessing import PROCESSED_PATH


MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42
N_FOLDS = 5
MAX_ROUNDS = 8000
EARLY_STOP = 100


CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type",
]


# Base params from Optuna Trial #17 (adapted from v4)
BASE_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
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


def run_experiment(name: str, X, y, learning_rate: float, max_rounds: int) -> dict:
    """
    Run a single experiment: 5-fold CV + final deployment model.
    Returns a dict with metrics and saves the model.
    """
    print("\n" + "=" * 60)
    print(f"Experiment: {name}")
    print(f"  learning_rate = {learning_rate}")
    print(f"  max_rounds    = {max_rounds}")
    print(f"  early_stop    = {EARLY_STOP}")
    print("=" * 60)

    params = {**BASE_PARAMS, "learning_rate": learning_rate}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, kss, ginis, best_iters = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature="auto")
        dval = lgb.Dataset(X_val, y_val, reference=dtrain, categorical_feature="auto")

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
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

    auc_mean = float(np.mean(aucs))
    auc_std = float(np.std(aucs))
    ks_mean = float(np.mean(kss))
    gini_mean = float(np.mean(ginis))
    best_iter_mean = int(np.mean(best_iters))

    print(f"\n  Summary -> AUC: {auc_mean:.4f} (+/- {auc_std:.4f})  "
          f"KS: {ks_mean:.4f}  Gini: {gini_mean:.4f}  avg_best_iter: {best_iter_mean}")

    # ---- Train deployment model on ALL data ----
    print(f"  Training deployment model on ALL data ...")
    final_rounds = int(best_iter_mean * 1.05)
    dtrain_all = lgb.Dataset(X, y, categorical_feature="auto")
    final_model = lgb.train(params, dtrain_all, num_boost_round=final_rounds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"lightgbm_{name}.joblib"
    joblib.dump(final_model, model_path)
    print(f"  Saved: {model_path}")
    print(f"  Trained with {final_rounds} rounds")

    return {
        "name": name,
        "learning_rate": learning_rate,
        "max_rounds": max_rounds,
        "auc": auc_mean,
        "auc_std": auc_std,
        "ks": ks_mean,
        "gini": gini_mean,
        "avg_best_iter": best_iter_mean,
        "fold_results": [
            {"fold": i + 1, "auc": float(aucs[i]), "ks": float(kss[i]),
             "best_iter": int(best_iters[i])}
            for i in range(N_FOLDS)
        ],
    }


def print_comparison(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("Ablation Study Results")
    print("=" * 60)
    header = f"{'Name':<10} {'lr':<8} {'max_rounds':<12} {'AUC':<10} {'KS':<10} {'avg_best_iter':<14}"
    print(header)
    print("-" * 64)
    for r in results:
        print(f"{r['name']:<10} {r['learning_rate']:<8} {r['max_rounds']:<12} "
              f"{r['auc']:<10.4f} {r['ks']:<10.4f} {r['avg_best_iter']:<14}")

    # Determine best
    best = max(results, key=lambda x: x["auc"])
    print("-" * 64)
    print(f"\nBest model: {best['name']} (AUC={best['auc']:.4f})")


def save_results(results: list[dict]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "ablation_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {path}")


def main() -> None:
    df = load_data()
    X, y = prepare_features(df)

    results = []

    # Experiment v5: lr=0.011, max_rounds=8000
    r_v5 = run_experiment("v5", X, y, learning_rate=0.011, max_rounds=MAX_ROUNDS)
    results.append(r_v5)

    # Experiment v6: lr=0.025, max_rounds=8000
    r_v6 = run_experiment("v6", X, y, learning_rate=0.025, max_rounds=MAX_ROUNDS)
    results.append(r_v6)

    # Print & save comparison
    print_comparison(results)
    save_results(results)

    print("\nDone.")


if __name__ == "__main__":
    main()