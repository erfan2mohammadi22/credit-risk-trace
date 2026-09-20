"""
eda.py
------
Exploratory Data Analysis for the cleaned Lending Club dataset.
Creates and saves key plots to reports/figures/.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import PROCESSED_PATH


FIGURES_DIR = Path("reports/figures")


def load_clean_data(path: Path = PROCESSED_PATH) -> pd.DataFrame:
    print(f"Loading cleaned data from {path} ...")
    df = pd.read_parquet(path)
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    return df


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target"].value_counts().sort_index()
    labels = ["No default (0)", "Default (1)"]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(labels, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}\n({v / len(df) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Target distribution", fontsize=13)
    ax.set_ylabel("Number of loans")
    ax.set_ylim(0, counts.max() * 1.15)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=120)
    plt.close()
    print("Saved: 01_target_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    features = [
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "fico_range_low",
        "revol_util",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, col in enumerate(features):
        ax = axes[i]
        data = df[[col, "target"]].dropna()
        upper = data[col].quantile(0.99)
        data = data[data[col] <= upper]
        sns.histplot(
            data=data, x=col, hue="target",
            bins=40, ax=ax, palette=["#2ecc71", "#e74c3c"],
            stat="density", common_norm=False, alpha=0.6,
        )
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        if ax.legend_:
            ax.legend_.set_title("target")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_numeric_distributions.png", dpi=120)
    plt.close()
    print("Saved: 02_numeric_distributions.png")


def plot_categorical_features(df: pd.DataFrame) -> None:
    features = ["grade", "home_ownership", "purpose", "term", "verification_status"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    for i, col in enumerate(features):
        ax = axes[i]
        rates = df.groupby(col)["target"].agg(["mean", "count"])
        rates = rates[rates["count"] >= 100].sort_values("mean", ascending=False)
        ax.barh(rates.index.astype(str), rates["mean"] * 100, color="#3498db")
        for j, (idx, row) in enumerate(rates.iterrows()):
            ax.text(row["mean"] * 100, j,
                    f" {row['mean'] * 100:.1f}% (n={int(row['count']):,})",
                    va="center", fontsize=8)
        ax.set_title(f"Default rate by {col}", fontsize=11)
        ax.set_xlabel("Default rate (%)")
        ax.invert_yaxis()
    axes[-1].axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_categorical_features.png", dpi=120)
    plt.close()
    print("Saved: 03_categorical_features.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric = df.select_dtypes(include="number")
    numeric = numeric.loc[:, numeric.isnull().mean() < 0.5]
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, vmin=-1, vmax=1, square=True,
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 7},
    )
    ax.set_title("Correlation heatmap (numeric features)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_correlation_heatmap.png", dpi=120)
    plt.close()
    print("Saved: 04_correlation_heatmap.png")


def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("EDA summary")
    print("=" * 60)
    print(f"\nRows: {df.shape[0]:,}")
    print(f"Default rate: {df['target'].mean() * 100:.2f}%")

    print("\nDefault rate by grade:")
    gr = df.groupby("grade")["target"].agg(["mean", "count"])
    gr["mean"] = (gr["mean"] * 100).round(2)
    print(gr.to_string())

    print("\nDefault rate by term:")
    tm = df.groupby("term")["target"].agg(["mean", "count"])
    tm["mean"] = (tm["mean"] * 100).round(2)
    print(tm.to_string())

    print("\n" + "=" * 60)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_clean_data()

    sns.set_theme(style="whitegrid", context="notebook")

    plot_target_distribution(df)
    plot_numeric_distributions(df)
    plot_categorical_features(df)
    plot_correlation_heatmap(df)

    print_summary(df)

    print(f"\nAll figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()