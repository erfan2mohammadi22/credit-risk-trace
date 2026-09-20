"""
shap_analysis.py
----------------
Explain the final LightGBM model using SHAP.

Outputs:
- Global feature importance (bar plot)
- SHAP summary plot (beeswarm)
- Dependence plots for top features
- Local explanation for a single high-risk prediction
- Text summary of top features
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from preprocessing import PROCESSED_PATH


MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")
MODEL_PATH = MODELS_DIR / "lightgbm_final.joblib"
SAMPLE_SIZE = 20_000
RANDOM_STATE = 42


CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type",
]


def load_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    print(f"Loading data from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    return df


def load_model(path: Path = MODEL_PATH):
    print(f"Loading model from {path} ...")
    model = joblib.load(path)
    print("Model loaded.\n")
    return model


def prepare_features(df: pd.DataFrame):
    X = df.drop(columns=["target", "loan_status"], errors="ignore")
    y = df["target"].values
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X, y


def sample_for_shap(X: pd.DataFrame, y: np.ndarray):
    """Take a random sample for SHAP analysis (full data is too slow)."""
    print(f"Sampling {SAMPLE_SIZE:,} rows for SHAP analysis ...")
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=min(SAMPLE_SIZE, len(X)), replace=False)
    X_sample = X.iloc[idx].reset_index(drop=True)
    y_sample = y[idx]
    print(f"Sample shape: {X_sample.shape}\n")
    return X_sample, y_sample


def compute_shap_values(model: lgb.Booster, X_sample: pd.DataFrame):
    """Compute SHAP values using LightGBM's native TreeExplainer."""
    print("Computing SHAP values ...")
    print("(this may take 1 to 3 minutes)")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    # For binary classification, LightGBM returns a single array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    print(f"SHAP values shape: {shap_values.shape}\n")
    return explainer, shap_values


def plot_summary_beeswarm(shap_values, X_sample):
    print("Creating summary (beeswarm) plot ...")
    plt.figure(figsize=(10, 12))
    shap.summary_plot(
        shap_values, X_sample,
        show=False, max_display=20, plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 05_shap_summary.png")


def plot_summary_bar(shap_values, X_sample):
    print("Creating feature importance (bar) plot ...")
    plt.figure(figsize=(10, 10))
    shap.summary_plot(
        shap_values, X_sample,
        plot_type="bar", show=False, max_display=20, plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 06_shap_importance.png")


def get_top_features(shap_values, X_sample, top_n: int = 10) -> list:
    """Return the names of the top-N most important features."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X_sample.columns)
    return importance.sort_values(ascending=False).head(top_n).index.tolist()


def plot_dependence_plots(shap_values, X_sample, top_features: list):
    """Dependence plots for the top features."""
    print("Creating dependence plots ...")
    n = len(top_features)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, feat in enumerate(top_features):
        ax = axes[i]
        feat_idx = X_sample.columns.get_loc(feat)
        feat_values = X_sample[feat]
        # Convert categorical to codes for plotting
        if str(feat_values.dtype) == "category":
            feat_values = feat_values.cat.codes
        ax.scatter(feat_values, shap_values[:, feat_idx],
                   s=4, alpha=0.4, c=shap_values[:, feat_idx], cmap="coolwarm")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"SHAP dependence: {feat}", fontsize=10)
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_shap_dependence.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Saved: 07_shap_dependence.png")


def plot_local_explanation(model, X_sample, y_sample, shap_values, explainer):
    """Waterfall plot for a single high-risk prediction."""
    print("Creating local explanation (waterfall) plot ...")

    # Pick a real defaulter with high predicted probability
    y_proba = model.predict(X_sample)
    candidates = np.where((y_sample == 1) & (y_proba > 0.7))[0]
    if len(candidates) == 0:
        print("  No high-risk defaulters found in sample. Skipping.")
        return
    idx = candidates[0]

    # Build a SHAP Explanation object for the waterfall plot
    explanation = shap.Explanation(
        values=shap_values[idx],
        base_values=explainer.expected_value
        if np.isscalar(explainer.expected_value)
        else explainer.expected_value[1],
        data=X_sample.iloc[idx].values,
        feature_names=X_sample.columns.tolist(),
    )
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_shap_local.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: 08_shap_local.png (row index {idx}, PD={y_proba[idx]:.3f})")


def print_top_features(shap_values, X_sample, top_n: int = 15):
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=X_sample.columns)
    importance = importance.sort_values(ascending=False).head(top_n)
    print("\n" + "=" * 60)
    print(f"Top {top_n} most important features (by mean |SHAP|)")
    print("=" * 60)
    for i, (feat, val) in enumerate(importance.items(), 1):
        print(f"  {i:2d}. {feat:<30s} {val:.4f}")
    print("=" * 60)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = prepare_features(df)
    model = load_model()

    X_sample, y_sample = sample_for_shap(X, y)
    explainer, shap_values = compute_shap_values(model, X_sample)

    # Figures
    plot_summary_beeswarm(shap_values, X_sample)
    plot_summary_bar(shap_values, X_sample)

    top_features = get_top_features(shap_values, X_sample, top_n=8)
    plot_dependence_plots(shap_values, X_sample, top_features)
    plot_local_explanation(model, X_sample, y_sample, shap_values, explainer)

    # Text summary
    print_top_features(shap_values, X_sample, top_n=15)

    print(f"\nAll SHAP figures saved to {FIGURES_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()