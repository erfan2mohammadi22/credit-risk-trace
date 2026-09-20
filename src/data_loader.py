"""
data_loader.py
--------------
Load and inspect the Lending Club dataset.

This module:
- Loads only the key columns (to save memory)
- Builds a binary target variable (default / no default) from loan_status
- Prints a summary of the dataset
"""

from pathlib import Path

import pandas as pd


# Default dataset path (gzip-compressed CSV)
DATA_PATH = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")


# Key columns needed for modeling
KEY_COLUMNS = [
    # Target source
    "loan_status",
    # Loan information
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "purpose",
    "application_type",
    # Borrower information
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "addr_state",
    "dti",
    # Credit history
    "delinq_2yrs",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    # Dates
    "issue_d",
    "earliest_cr_line",
]


# Mapping loan_status to a binary target
# 0 = no default, 1 = default
DEFAULT_STATUSES = {
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Charged Off",
}

NON_DEFAULT_STATUSES = {
    "Fully Paid",
    "Does not meet the credit policy. Status:Fully Paid",
}


def load_data(
    path: Path = DATA_PATH,
    usecols: list | None = None,
) -> pd.DataFrame:
    """
    Load the Lending Club dataset from a gzip-compressed CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download it from Kaggle and place it in data/raw/."
        )

    print(f"Loading dataset from {path} ...")
    print("(this may take 1 to 3 minutes)")

    df = pd.read_csv(
        path,
        usecols=usecols,
        low_memory=False,
    )

    print(f"Load complete: {df.shape[0]:,} rows, {df.shape[1]} columns\n")
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a binary target variable from loan_status.

    - 0: no default (Fully Paid)
    - 1: default (Charged Off / Default)
    """
    print("Building binary target from loan_status ...")

    print("\nOriginal loan_status distribution:")
    print(df["loan_status"].value_counts().to_string())

    def map_status(status: str) -> int | None:
        if status in NON_DEFAULT_STATUSES:
            return 0
        if status in DEFAULT_STATUSES:
            return 1
        return None

    df = df.copy()
    df["target"] = df["loan_status"].map(map_status)

    before = len(df)
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    after = len(df)
    print(f"\nDropped {before - after:,} rows without a final status.")
    print(f"Remaining: {after:,} rows")

    df["target"] = df["target"].astype(int)
    return df


def summarize(df: pd.DataFrame, target_col: str = "target") -> None:
    """
    Print a summary of the dataset.
    """
    print("\n" + "=" * 60)
    print("Dataset summary")
    print("=" * 60)

    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nMissing values (only columns with missing values):")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        "count": missing,
        "percent": missing_pct,
    })
    missing_df = missing_df[missing_df["count"] > 0].sort_values(
        "percent", ascending=False
    )
    if missing_df.empty:
        print("  No missing values.")
    else:
        print(missing_df.to_string())

    if target_col in df.columns:
        print(f"\nTarget distribution '{target_col}':")
        counts = df[target_col].value_counts()
        pcts = df[target_col].value_counts(normalize=True) * 100
        for val in sorted(counts.index):
            label = "default" if val == 1 else "no default"
            print(f"  {val} ({label}): {counts[val]:,} ({pcts[val]:.2f}%)")

    print("\nDescriptive statistics (numeric columns):")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        print(numeric_df.describe().T.to_string())

    print("\n" + "=" * 60)


def main() -> None:
    """
    Main entry point when the file is run directly.
    """
    df = load_data(usecols=KEY_COLUMNS)
    df = create_target(df)
    summarize(df)


if __name__ == "__main__":
    main()