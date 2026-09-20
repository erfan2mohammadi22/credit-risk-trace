"""
expected_loss.py
----------------
Compute Expected Loss (EL) for each loan and perform risk-based pricing.

Formula:
    EL = PD × LGD × EAD

Where:
    PD  = Probability of Default (from LightGBM model)
    LGD = Loss Given Default (industry standard ~45%)
    EAD = Exposure at Default (loan amount)

Also computes:
    - Risk-based interest rate
    - Portfolio-level EL statistics
    - Visualizations
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocessing import PROCESSED_PATH


# ---------- Configuration ----------
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
MODEL_PATH = MODELS_DIR / "lightgbm_final.joblib"

# Financial assumptions (industry standard)
LGD = 0.45              # Loss Given Default: 45% of exposure is lost
COST_OF_FUNDS = 0.02    # 2% cost of capital
OPERATING_COST = 0.01   # 1% operational cost
PROFIT_MARGIN = 0.03    # 3% target profit margin

CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type",
]


# ---------- Data loading ----------
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


# ---------- Core computations ----------
def compute_pd(model, X: pd.DataFrame) -> np.ndarray:
    """Predict Probability of Default for every row."""
    print("Computing PD for all loans ...")
    pd_array = model.predict(X)
    print(f"  PD range: [{pd_array.min():.4f}, {pd_array.max():.4f}]")
    print(f"  PD mean:  {pd_array.mean():.4f}\n")
    return pd_array


def compute_el(df: pd.DataFrame, pd_array: np.ndarray) -> pd.DataFrame:
    """Add PD, EAD, EL, and risk-based rate columns."""
    df = df.copy()

    # EAD = loan amount (simplification for demo)
    df["pd"] = pd_array
    df["lgd"] = LGD
    df["ead"] = df["loan_amnt"]

    # EL = PD × LGD × EAD
    df["expected_loss"] = df["pd"] * df["lgd"] * df["ead"]

    # Expected loss rate (EL / EAD)
    df["el_rate"] = df["pd"] * df["lgd"]

    # Risk-based pricing:
    # minimum_rate = cost_of_funds + el_rate + operating_cost + profit_margin
    df["min_rate"] = (
        COST_OF_FUNDS + df["el_rate"] + OPERATING_COST + PROFIT_MARGIN
    ) * 100  # to percentage

    return df


# ---------- Visualizations ----------
def plot_pd_distribution(df: pd.DataFrame) -> None:
    print("Creating PD distribution plot ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Overall PD distribution
    axes[0].hist(df["pd"], bins=50, color="#3498db", edgecolor="white")
    axes[0].axvline(df["pd"].mean(), color="red", linestyle="--",
                    label=f"Mean = {df['pd'].mean():.3f}")
    axes[0].set_title("PD distribution (all loans)")
    axes[0].set_xlabel("Probability of Default")
    axes[0].set_ylabel("Number of loans")
    axes[0].legend()

    # PD by actual target
    for label, color, name in [(0, "#2ecc71", "No default"),
                                (1, "#e74c3c", "Default")]:
        subset = df[df["target"] == label]["pd"]
        axes[1].hist(subset, bins=50, alpha=0.6, color=color,
                     label=name, edgecolor="white")
    axes[1].set_title("PD distribution by actual outcome")
    axes[1].set_xlabel("Probability of Default")
    axes[1].set_ylabel("Number of loans")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_pd_distribution.png", dpi=120)
    plt.close()
    print("  Saved: 09_pd_distribution.png")


def plot_el_by_grade(df: pd.DataFrame) -> None:
    print("Creating EL by grade plot ...")
    summary = df.groupby("grade").agg(
        mean_pd=("pd", "mean"),
        mean_el=("expected_loss", "mean"),
        mean_rate=("min_rate", "mean"),
        count=("pd", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Mean PD by grade
    axes[0].bar(summary["grade"], summary["mean_pd"] * 100, color="#e74c3c")
    axes[0].set_title("Mean PD by grade")
    axes[0].set_ylabel("PD (%)")
    for i, v in enumerate(summary["mean_pd"] * 100):
        axes[0].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    # Mean EL by grade (in dollars)
    axes[1].bar(summary["grade"], summary["mean_el"], color="#3498db")
    axes[1].set_title("Mean Expected Loss by grade")
    axes[1].set_ylabel("EL ($)")
    for i, v in enumerate(summary["mean_el"]):
        axes[1].text(i, v + 20, f"${v:.0f}", ha="center", fontsize=9)

    # Minimum rate by grade
    axes[2].bar(summary["grade"], summary["mean_rate"], color="#f39c12")
    axes[2].set_title("Risk-based minimum rate by grade")
    axes[2].set_ylabel("Rate (%)")
    for i, v in enumerate(summary["mean_rate"]):
        axes[2].text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_el_by_grade.png", dpi=120)
    plt.close()
    print("  Saved: 10_el_by_grade.png")


def plot_pd_vs_rate(df: pd.DataFrame) -> None:
    print("Creating PD vs rate plot ...")
    fig, ax = plt.subplots(figsize=(10, 6))

    sample = df.sample(n=min(10_000, len(df)), random_state=42)
    scatter = ax.scatter(
        sample["pd"], sample["min_rate"],
        c=sample["loan_amnt"], cmap="viridis",
        s=8, alpha=0.5,
    )
    plt.colorbar(scatter, label="Loan amount ($)")
    ax.set_xlabel("Probability of Default (PD)")
    ax.set_ylabel("Minimum rate (%)")
    ax.set_title("Risk-based pricing: PD vs minimum rate")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "11_pd_vs_rate.png", dpi=120)
    plt.close()
    print("  Saved: 11_pd_vs_rate.png")


# ---------- Summary ----------
def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("Expected Loss Summary")
    print("=" * 60)

    total_loans = len(df)
    total_ead = df["ead"].sum()
    total_el = df["expected_loss"].sum()
    avg_pd = df["pd"].mean()
    avg_el = df["expected_loss"].mean()
    avg_rate = df["min_rate"].mean()

    print(f"\nPortfolio size: {total_loans:,} loans")
    print(f"Total exposure (EAD): ${total_ead:,.0f}")
    print(f"Average PD: {avg_pd:.4f} ({avg_pd * 100:.2f}%)")
    print(f"Average EL per loan: ${avg_el:,.2f}")
    print(f"Average minimum rate: {avg_rate:.2f}%")

    print(f"\nTotal expected loss: ${total_el:,.0f}")
    print(f"Expected loss rate: {total_el / total_ead * 100:.2f}%")

    print("\n" + "-" * 60)
    print("Breakdown by grade:")
    print("-" * 60)
    summary = df.groupby("grade").agg(
        n_loans=("pd", "count"),
        mean_pd=("pd", "mean"),
        mean_el=("expected_loss", "mean"),
        mean_rate=("min_rate", "mean"),
    ).round(4)
    summary["mean_pd"] = (summary["mean_pd"] * 100).round(2)
    summary["mean_el"] = summary["mean_el"].round(0)
    summary["mean_rate"] = summary["mean_rate"].round(2)
    print(summary.to_string())

    print("\n" + "=" * 60)


def save_results(df: pd.DataFrame) -> None:
    """Save enriched dataset for later use in HTML deployment."""
    out_path = Path("data/processed/loans_with_el.parquet")
    cols = [
        "loan_amnt", "int_rate", "term", "grade", "sub_grade",
        "annual_inc", "dti", "fico_range_low",
        "target", "pd", "lgd", "ead", "expected_loss", "el_rate", "min_rate",
    ]
    out_cols = [c for c in cols if c in df.columns]
    df[out_cols].to_parquet(out_path, index=False)
    print(f"\nSaved enriched data to: {out_path}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, _ = prepare_features(df)
    model = load_model()

    pd_array = compute_pd(model, X)
    df = compute_el(df, pd_array)

    plot_pd_distribution(df)
    plot_el_by_grade(df)
    plot_pd_vs_rate(df)

    print_summary(df)
    save_results(df)

    print("\nDone.")


if __name__ == "__main__":
    main()