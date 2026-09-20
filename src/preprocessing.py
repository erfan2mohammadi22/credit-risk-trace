"""
preprocessing.py
----------------
Clean and preprocess the Lending Club dataset with advanced features.

Version: v4 (current)
Previous: v3 (with leakage: issue_year, addr_state)

Changes from v3:
- Removed issue_year, issue_month, issue_quarter (temporal leakage)
- Removed addr_state (fairness concerns + noisy signal)
- See DECISIONS.md for full rationale
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import KEY_COLUMNS, create_target, load_data


PROCESSED_PATH = Path("data/processed/loans_clean.parquet")


def fix_term(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return series.str.extract(r"(\d+)").astype(int).squeeze()


def fix_int_rate(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return series.str.replace("%", "", regex=False).astype(float)


def fix_emp_length(series: pd.Series) -> pd.Series:
    mapping = {
        "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
        "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
        "8 years": 8, "9 years": 9, "10+ years": 10,
    }
    return series.map(mapping)


def fix_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning dataset ...")
    df = df.copy()

    # ---- Fix data types ----
    print("  - Fixing data types")
    df["term"] = fix_term(df["term"])
    df["int_rate"] = fix_int_rate(df["int_rate"])
    df["emp_length"] = fix_emp_length(df["emp_length"])
    df["issue_d"] = fix_date(df["issue_d"])
    df["earliest_cr_line"] = fix_date(df["earliest_cr_line"])

    # ---- Missing indicators (before imputing) ----
    print("  - Adding missing indicators")
    for col in ["mths_since_last_delinq", "emp_length", "mort_acc"]:
        df[f"{col}_missing"] = df[col].isnull().astype(int)

    # ---- Handle non-sensible values ----
    print("  - Handling non-sensible values")
    df.loc[df["dti"] < 0, "dti"] = np.nan
    df.loc[df["dti"] > 100, "dti"] = np.nan
    df.loc[df["annual_inc"] <= 0, "annual_inc"] = np.nan
    df.loc[df["revol_util"] > 150, "revol_util"] = np.nan

    # ---- Time-based features ----
    # NOTE: Only credit_history_months is kept because it is derived
    # from the borrower's own history, not from the calendar date.
    # issue_year/month/quarter were removed after SHAP analysis
    # revealed they caused temporal leakage (see DECISIONS.md).
    print("  - Computing credit history")
    df["credit_history_months"] = (
        (df["issue_d"] - df["earliest_cr_line"]).dt.days / 30.44
    ).round(0)
    df.loc[df["credit_history_months"] < 0, "credit_history_months"] = np.nan

    # ---- Basic ratios ----
    print("  - Computing basic ratios")
    df["loan_income_ratio"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["monthly_income"] = df["annual_inc"] / 12
    df["installment_to_income"] = df["installment"] / (df["monthly_income"] + 1)
    df["balance_per_account"] = df["revol_bal"] / (df["open_acc"] + 1)
    df["delinq_per_account"] = df["delinq_2yrs"] / (df["total_acc"] + 1)

    # ---- Log transforms ----
    print("  - Applying log transforms")
    df["log_annual_inc"] = np.log1p(df["annual_inc"])
    df["log_revol_bal"] = np.log1p(df["revol_bal"])
    df["log_loan_amnt"] = np.log1p(df["loan_amnt"])

    # ---- Credit score features ----
    print("  - Computing FICO features")
    df["fico_avg"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["fico_range"] = df["fico_range_high"] - df["fico_range_low"]

    # ---- Aggregate features ----
    print("  - Computing aggregate features")
    df["total_credit_lines"] = df["open_acc"] + df["mort_acc"].fillna(0)
    df["has_delinq"] = (df["delinq_2yrs"] > 0).astype(int)
    df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)

    # ---- Drop leaky / derived columns ----
    print("  - Dropping leaky columns")
    df = df.drop(columns=["funded_amnt", "installment"], errors="ignore")

    # ---- Drop temporal and geographic features ----
    # Rationale:
    # - issue_d/earliest_cr_line: only used to build credit_history_months
    # - addr_state: high SHAP variance (noisy) + fairness concerns
    # - issue_year/month/quarter: temporal leakage (learned macro conditions)
    print("  - Dropping temporal and geographic features")
    df = df.drop(
        columns=[
            "issue_d",
            "earliest_cr_line",
            "addr_state",
        ],
        errors="ignore",
    )

    print(f"Cleaning complete. Shape: {df.shape}\n")
    return df


def save_parquet(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving cleaned data to {path} ...")
    df.to_parquet(path, index=False)
    print("Save complete.\n")


def main() -> None:
    df = load_data(usecols=KEY_COLUMNS)
    df = create_target(df)
    df = clean(df)
    save_parquet(df)


if __name__ == "__main__":
    main()