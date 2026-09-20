"""
export_onnx.py
--------------
Convert the final LightGBM model (v5) to ONNX format.

Steps:
1. Load the LightGBM model and data
2. Build category mappings (A→0, B→1, ...) for categorical features
3. Prepare TWO versions of X:
   - X_cat: with pandas 'category' dtype (for LightGBM)
   - X_enc: with integer encoding (for ONNX)
4. Convert to ONNX with onnxmltools
5. Verify ONNX predictions match LightGBM
6. Save model.onnx and metadata.json for the browser

Output:
- docs/onnx/model.onnx
- docs/onnx/metadata.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxmltools
import onnxruntime as rt
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType
from sklearn.metrics import roc_auc_score

from preprocessing import PROCESSED_PATH


# ---------- Paths ----------
MODELS_DIR = Path("models")
ONNX_DIR = Path("docs/onnx")
MODEL_PATH = MODELS_DIR / "lightgbm_final.joblib"
ONNX_PATH = ONNX_DIR / "model.onnx"
METADATA_PATH = ONNX_DIR / "metadata.json"


# ---------- Configuration ----------
CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type",
]

RANDOM_STATE = 42
TEST_SIZE = 5000


# ---------- Load ----------
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


# ---------- Category mappings ----------
def fit_category_mappings(df: pd.DataFrame) -> dict:
    """
    Build category -> code mappings for every categorical feature.
    These mappings are saved in metadata.json so the browser
    can encode user input the same way.
    """
    mappings = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            categories = df[col].astype("category").cat.categories.tolist()
            mappings[col] = {str(cat): i for i, cat in enumerate(categories)}
    return mappings


def encode_categoricals(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    """
    Replace categorical columns with their integer codes.
    Unseen categories become -1.
    """
    df = df.copy()
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].astype(str).map(mapping).fillna(-1).astype("int32")
    return df


def prepare_features(df: pd.DataFrame, mappings: dict):
    """
    Return TWO versions of X:
    - X_cat: with pandas 'category' dtype (for LightGBM prediction)
    - X_enc: with integer encoding as float32 (for ONNX)
    """
    X_base = df.drop(columns=["target", "loan_status"], errors="ignore")
    y = df["target"].values

    # Version 1: for LightGBM (keep categories as category dtype)
    X_cat = X_base.copy()
    for col in CATEGORICAL_COLS:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype("category")

    # Version 2: for ONNX (integer codes as float32)
    X_enc = encode_categoricals(X_base, mappings)
    X_enc = X_enc.astype("float32")

    return X_cat, X_enc, y


# ---------- ONNX conversion ----------
def convert_to_onnx(model, X_sample: pd.DataFrame) -> onnx.ModelProto:
    """Convert LightGBM Booster to ONNX."""
    print("Converting LightGBM model to ONNX ...")

    n_features = X_sample.shape[1]

    initial_types = [("input", FloatTensorType([None, n_features]))]

    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_types,
        name="credit_risk_lgbm",
        target_opset=15,
        zipmap=False,
    )
    print("Conversion complete.\n")
    return onnx_model


# ---------- Verification ----------
def verify_onnx(
    onnx_path: Path,
    model,
    X_cat_test: pd.DataFrame,
    X_enc_test: pd.DataFrame,
    y_test: np.ndarray,
):
    """Check that ONNX predictions match LightGBM predictions."""
    print("Verifying ONNX vs LightGBM ...")

    # LightGBM predictions (uses categorical dtype)
    lgb_pred = model.predict(X_cat_test)

    # ONNX predictions (uses numeric encoding)
    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input = {"input": X_enc_test.values.astype(np.float32)}
    onnx_out = sess.run(None, onnx_input)

    # Outputs: [label, probabilities] where probabilities is (n, 2)
    onnx_prob = onnx_out[1][:, 1]

    # Compare
    max_diff = float(np.max(np.abs(lgb_pred - onnx_prob)))
    print(f"  Max absolute difference: {max_diff:.8f}")

    if max_diff < 1e-4:
        print("  ONNX and LightGBM predictions match!\n")
    else:
        print("  WARNING: predictions differ. Check conversion.\n")

    auc_lgb = roc_auc_score(y_test, lgb_pred)
    auc_onnx = roc_auc_score(y_test, onnx_prob)
    print(f"  LightGBM AUC: {auc_lgb:.6f}")
    print(f"  ONNX     AUC: {auc_onnx:.6f}")
    print(f"  Difference:   {abs(auc_lgb - auc_onnx):.8f}\n")

    return max_diff


# ---------- Metadata ----------
def save_metadata(
    feature_names: list,
    mappings: dict,
    X_sample: pd.DataFrame,
    onnx_path: Path,
    metadata_path: Path,
    max_diff: float,
):
    """Save metadata for the browser (feature order, ranges, mappings)."""
    ranges = {}
    for col in feature_names:
        if col in X_sample.columns:
            ranges[col] = {
                "min": float(X_sample[col].min()),
                "max": float(X_sample[col].max()),
                "mean": float(X_sample[col].mean()),
                "median": float(X_sample[col].median()),
            }

    metadata = {
        "model_file": onnx_path.name,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "categorical_features": list(mappings.keys()),
        "category_mappings": mappings,
        "feature_ranges": ranges,
        "verification": {
            "max_abs_diff_lgb_vs_onnx": max_diff,
            "verified": bool(max_diff < 1e-4),
        },
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_path}\n")


# ---------- Main ----------
def main() -> None:
    ONNX_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    model = load_model()

    # Fit category mappings from FULL data
    print("Fitting category mappings ...")
    mappings = fit_category_mappings(df)
    print(f"  Categorical features: {list(mappings.keys())}")
    for col, mapping in mappings.items():
        print(f"    {col}: {len(mapping)} categories")
    print()

    # Prepare both feature versions
    X_cat, X_enc, y = prepare_features(df, mappings)
    print(f"Feature matrix: {X_enc.shape}")
    print(f"Feature names: {list(X_enc.columns)}\n")

    # Random sample to test
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_enc), size=TEST_SIZE, replace=False)
    X_enc_test = X_enc.iloc[idx].reset_index(drop=True)
    X_cat_test = X_cat.iloc[idx].reset_index(drop=True)
    y_test = y[idx]

    # Convert to ONNX (uses numeric-encoded sample)
    onnx_model = convert_to_onnx(model, X_enc_test)

    # Save ONNX
    onnx.save_model(onnx_model, str(ONNX_PATH))
    print(f"Saved ONNX model: {ONNX_PATH}")
    print(f"  File size: {ONNX_PATH.stat().st_size / 1024:.1f} KB\n")

    # Verify
    max_diff = verify_onnx(ONNX_PATH, model, X_cat_test, X_enc_test, y_test)

    # Save metadata
    save_metadata(
        feature_names=list(X_enc.columns),
        mappings=mappings,
        X_sample=X_enc,
        onnx_path=ONNX_PATH,
        metadata_path=METADATA_PATH,
        max_diff=max_diff,
    )

    print("Done.")


if __name__ == "__main__":
    main()