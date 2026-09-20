"""
export_presets.py
-----------------
Pre-compute predictions + SHAP values for the 5 demo presets.

Why: the browser demo currently requires downloading a 30 MB ONNX model
and initializing a heavy runtime before showing any result. That's fine
for power users, but painful for casual visitors.

This script lets the demo show instant results for the 5 preset profiles
by pre-computing them here (once, offline) and saving to presets.json.

The model is the same v5 LightGBM — the numbers are identical to what
the browser would compute. AUC is unaffected.

Output:
    docs/onnx/presets.json
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


# ---------- Paths ----------
MODELS_DIR = Path("models")
ONNX_DIR = Path("docs/onnx")
MODEL_PATH = MODELS_DIR / "lightgbm_final.joblib"
OUTPUT_PATH = ONNX_DIR / "presets.json"


# ---------- Config ----------
CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type",
]

# Financial assumptions (identical to expected_loss.py and demo.js)
LGD = 0.45
COST_OF_FUNDS = 0.02
OPERATING_COST = 0.01
PROFIT_MARGIN = 0.03

# Top N features to include in the SHAP chart
TOP_SHAP_FEATURES = 6


# ---------- Presets (mirrors docs/js/demo.js) ----------
PRESETS = {
    "excellent": {
        "name": "Excellent",
        "emoji": "🌟",
        "inputs": {
            "loan_amnt": 10000, "term": 36, "int_rate": 6.5,
            "grade": "A", "sub_grade": "A1",
            "annual_inc": 150000, "dti": 5, "fico_range_low": 810,
            "home_ownership": "MORTGAGE", "purpose": "debt_consolidation",
            "emp_length": 10, "revol_util": 10,
        },
    },
    "good": {
        "name": "Good",
        "emoji": "🟢",
        "inputs": {
            "loan_amnt": 12000, "term": 36, "int_rate": 10.5,
            "grade": "B", "sub_grade": "B2",
            "annual_inc": 85000, "dti": 12, "fico_range_low": 740,
            "home_ownership": "MORTGAGE", "purpose": "credit_card",
            "emp_length": 7, "revol_util": 30,
        },
    },
    "average": {
        "name": "Average",
        "emoji": "🟡",
        "inputs": {
            "loan_amnt": 15000, "term": 36, "int_rate": 14.5,
            "grade": "C", "sub_grade": "C3",
            "annual_inc": 60000, "dti": 20, "fico_range_low": 690,
            "home_ownership": "RENT", "purpose": "debt_consolidation",
            "emp_length": 4, "revol_util": 55,
        },
    },
    "risky": {
        "name": "Risky",
        "emoji": "🟠",
        "inputs": {
            "loan_amnt": 20000, "term": 60, "int_rate": 20.5,
            "grade": "E", "sub_grade": "E3",
            "annual_inc": 45000, "dti": 28, "fico_range_low": 655,
            "home_ownership": "RENT", "purpose": "small_business",
            "emp_length": 2, "revol_util": 75,
        },
    },
    "very_risky": {
        "name": "Very Risky",
        "emoji": "🔴",
        "inputs": {
            "loan_amnt": 25000, "term": 60, "int_rate": 26.5,
            "grade": "F", "sub_grade": "F4",
            "annual_inc": 32000, "dti": 35, "fico_range_low": 640,
            "home_ownership": "RENT", "purpose": "small_business",
            "emp_length": 1, "revol_util": 92,
        },
    },
}


# ---------- Helpers ----------
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print(f"Loading model: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"  Features: {len(model.feature_name())}")
    print(f"  Trees:    {model.num_trees()}")
    return model


def build_feature_row(preset_inputs: dict, feature_names: list) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the exact order the model expects.

    All derived features here must match:
      - preprocessing.py (Python pipeline)
      - onnx-runner.js computeDerivedFeatures() (browser)
    """
    d = dict(preset_inputs)

    # --- Implicit inputs (demo form defaults) ---
    d["fico_range_high"] = d["fico_range_low"] + 4
    d["delinq_2yrs"] = 0
    d["inq_last_6mths"] = 0
    d["mths_since_last_delinq"] = 60
    d["open_acc"] = 10
    d["pub_rec"] = 0
    d["total_acc"] = 20
    d["application_type"] = "Individual"
    d["mort_acc"] = 2 if d["home_ownership"] == "MORTGAGE" else 0
    d["pub_rec_bankruptcies"] = 0
    d["credit_history_months"] = 120
    d["verification_status"] = "Verified"

    # --- Revoling balance estimated from utilization ---
    d["revol_bal"] = round((d["revol_util"] / 100) * d["annual_inc"] * 0.3)

    # --- Installment from loan amount + rate + term ---
    r = d["int_rate"] / 100 / 12
    n = d["term"]
    d["installment"] = round(d["loan_amnt"] * r / (1 - (1 + r) ** (-n)))

    # --- Derived ratios / logs / FICO ---
    monthly_income = d["annual_inc"] / 12
    d["loan_income_ratio"] = d["loan_amnt"] / (d["annual_inc"] + 1)
    d["monthly_income"] = monthly_income
    d["installment_to_income"] = d["installment"] / (monthly_income + 1)
    d["balance_per_account"] = d["revol_bal"] / (d["open_acc"] + 1)
    d["delinq_per_account"] = d["delinq_2yrs"] / (d["total_acc"] + 1)
    d["log_annual_inc"] = float(np.log1p(d["annual_inc"]))
    d["log_revol_bal"] = float(np.log1p(d["revol_bal"]))
    d["log_loan_amnt"] = float(np.log1p(d["loan_amnt"]))
    d["fico_avg"] = (d["fico_range_low"] + d["fico_range_high"]) / 2
    d["fico_range"] = d["fico_range_high"] - d["fico_range_low"]
    d["total_credit_lines"] = d["open_acc"] + d["mort_acc"]
    d["has_delinq"] = 1 if d["delinq_2yrs"] > 0 else 0
    d["has_bankruptcy"] = 1 if d["pub_rec_bankruptcies"] > 0 else 0
    d["mths_since_last_delinq_missing"] = 0
    d["emp_length_missing"] = 0
    d["mort_acc_missing"] = 0

    # --- Keep only model features, in order ---
    row = {k: d.get(k, -1) for k in feature_names}
    df = pd.DataFrame([row])

    # Cast categoricals so LightGBM treats them properly
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def pd_to_credit_score(pd_value: float) -> int:
    """
    Mirror the JS formula in demo.js.
    Higher PD → lower score.
    """
    log_odds = np.log(pd_value / (1 - pd_value) + 0.001)
    score = 850 - 80 * (log_odds + 4)
    score = max(300, min(850, score))
    return int(round(score))


def compute_shap_top(
    explainer: shap.TreeExplainer,
    X_row: pd.DataFrame,
    feature_names: list,
    top_n: int,
) -> list:
    """
    Compute SHAP for a single row, return top-N by absolute impact.
    """
    shap_vals = explainer.shap_values(X_row)

    # For LightGBM binary classifier, shap_values returns array of shape (n, features)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    shap_vals = np.asarray(shap_vals).reshape(-1)

    # Pair each feature with its value and impact
    pairs = []
    for i, name in enumerate(feature_names):
        raw = X_row.iloc[0][name]
        if isinstance(raw, (np.floating, float)):
            display = round(float(raw), 3)
        else:
            display = str(raw)
        pairs.append({
            "feature": name,
            "value": display,
            "impact": float(shap_vals[i]),
        })

    # Sort by |impact| descending
    pairs.sort(key=lambda p: abs(p["impact"]), reverse=True)
    return pairs[:top_n]


# ---------- Main ----------
def main():
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    feature_names = model.feature_name()

    print("\nInitializing SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    print("  Ready.\n")

    results = {}

    for key, preset in PRESETS.items():
        print(f"── {preset['name']} ({key}) ──")
        X_row = build_feature_row(preset["inputs"], feature_names)

        # PD
        pd_value = float(model.predict(X_row)[0])
        print(f"  PD:  {pd_value:.4f} ({pd_value * 100:.2f}%)")

        # Credit score (same formula as JS)
        score = pd_to_credit_score(pd_value)
        print(f"  Score: {score}")

        # Expected Loss
        el = pd_value * LGD * preset["inputs"]["loan_amnt"]
        el_rate = pd_value * LGD
        min_rate = (COST_OF_FUNDS + el_rate + OPERATING_COST + PROFIT_MARGIN) * 100
        print(f"  EL:  ${el:.2f}")
        print(f"  Min rate: {min_rate:.2f}%")

        # SHAP
        shap_top = compute_shap_top(explainer, X_row, feature_names, TOP_SHAP_FEATURES)
        print(f"  SHAP top {TOP_SHAP_FEATURES}:")
        for item in shap_top:
            sign = "+" if item["impact"] >= 0 else ""
            print(f"    {item['feature']:<28s} {sign}{item['impact']:.4f}")

        results[key] = {
            "name": preset["name"],
            "emoji": preset["emoji"],
            "inputs": preset["inputs"],
            "pd": round(pd_value, 6),
            "credit_score": score,
            "el": round(el, 2),
            "el_rate": round(el_rate, 6),
            "min_rate": round(min_rate, 2),
            "lgd": LGD,
            "shap_top": [
                {
                    "feature": item["feature"],
                    "value": item["value"],
                    "impact": round(item["impact"], 6),
                }
                for item in shap_top
            ],
        }
        print()

    # Meta block
    output = {
        "meta": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": "lightgbm_v5",
            "auc": 0.7287,
            "n_features": len(feature_names),
            "lgd": LGD,
            "cost_of_funds": COST_OF_FUNDS,
            "operating_cost": OPERATING_COST,
            "profit_margin": PROFIT_MARGIN,
        },
        "presets": results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print("=" * 60)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Size:  {size_kb:.2f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()