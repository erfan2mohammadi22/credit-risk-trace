"""
CreditTrace — Streamlit Edition (Complete)
------------------------------------------
The most comprehensive credit risk analysis tool.

Tabs:
  1. Custom Analysis — SHAP + Counterfactual + Export
  2. Portfolio Simulator — grade distribution → total EL
  3. Compare Borrowers — side-by-side A/B analysis
  4. Batch Prediction — CSV upload → predictions + download
  5. Model Insights — ROC, confusion matrix, cost analysis

All HTML rendered via st.html() to avoid markdown parsing issues.
"""

import base64
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st


# ============================================
# Constants
# ============================================
LGD = 0.45
COST_OF_FUNDS = 0.02
OPERATING_COST = 0.01
PROFIT_MARGIN = 0.03
AUC = 0.7287
KS = 0.3322

MODEL_PATH = Path(__file__).parent / "model.joblib"
METADATA_PATH = Path(__file__).parent / "metadata.json"

CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "application_type",
]

COLOR_BG = "#0a0e1a"
COLOR_CARD = "#141b2d"
COLOR_BORDER = "#1f2937"
COLOR_ACCENT = "#00d4ff"
COLOR_PURPLE = "#8b5cf6"
COLOR_SUCCESS = "#00e676"
COLOR_WARNING = "#ffab00"
COLOR_DANGER = "#ff5252"
COLOR_TEXT = "#e8eef5"
COLOR_MUTED = "#94a3b8"
COLOR_DIM = "#64748b"

GRADE_COLORS = {
    "A": "#00e676", "B": "#00c4b4", "C": "#00d4ff",
    "D": "#ffab00", "E": "#ff7b00", "F": "#ff5252", "G": "#d32f2f",
}

GRADE_PROFILES = {
    "A": {"int_rate": 7.5,  "fico": 810, "annual_inc": 110000, "dti": 10, "loan_amnt": 15000, "term": 36, "revol_util": 20, "emp_length": 8,  "home_ownership": "MORTGAGE", "purpose": "credit_card"},
    "B": {"int_rate": 11.0, "fico": 760, "annual_inc": 85000,  "dti": 15, "loan_amnt": 14000, "term": 36, "revol_util": 35, "emp_length": 6,  "home_ownership": "MORTGAGE", "purpose": "credit_card"},
    "C": {"int_rate": 14.5, "fico": 700, "annual_inc": 65000,  "dti": 20, "loan_amnt": 15000, "term": 36, "revol_util": 50, "emp_length": 5,  "home_ownership": "RENT",     "purpose": "debt_consolidation"},
    "D": {"int_rate": 18.0, "fico": 680, "annual_inc": 60000,  "dti": 25, "loan_amnt": 16000, "term": 36, "revol_util": 60, "emp_length": 4,  "home_ownership": "RENT",     "purpose": "debt_consolidation"},
    "E": {"int_rate": 21.0, "fico": 660, "annual_inc": 55000,  "dti": 28, "loan_amnt": 18000, "term": 60, "revol_util": 70, "emp_length": 3,  "home_ownership": "RENT",     "purpose": "small_business"},
    "F": {"int_rate": 24.0, "fico": 645, "annual_inc": 50000,  "dti": 32, "loan_amnt": 20000, "term": 60, "revol_util": 80, "emp_length": 2,  "home_ownership": "RENT",     "purpose": "small_business"},
    "G": {"int_rate": 27.0, "fico": 630, "annual_inc": 45000,  "dti": 35, "loan_amnt": 22000, "term": 60, "revol_util": 90, "emp_length": 1,  "home_ownership": "RENT",     "purpose": "small_business"},
}

DEFAULT_WEIGHTS = {"A": 20, "B": 25, "C": 22, "D": 15, "E": 10, "F": 5, "G": 3}


# ============================================
# Page config
# ============================================
st.set_page_config(
    page_title="CreditTrace — Risk Analysis Suite",
    page_icon="🔵",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================
# Custom CSS
# ============================================
st.html("""
<style>
    .stApp { background-color: #0a0e1a; }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    h1, h2, h3, h4 {
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: #e8eef5 !important;
        letter-spacing: -0.02em !important;
    }
    p, span, div, label, li {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #e8eef5;
    }

    .hero-wrap { text-align: center; margin-bottom: 1.5rem; }
    .hero-badge {
        display: inline-block;
        padding: 6px 14px;
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 100px;
        font-size: 0.72rem;
        color: #00d4ff;
        font-weight: 500;
        letter-spacing: 0.03em;
        margin-bottom: 0.9rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #e8eef5;
        letter-spacing: -0.03em;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .hero-title .grad {
        background: linear-gradient(135deg, #00d4ff, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.6;
    }

    .kpi-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 1.5rem 0;
    }
    .kpi-item {
        background: #141b2d;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 0.85rem;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.6rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 500;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #00d4ff;
        margin-top: 0.35rem;
        line-height: 1;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e8eef5;
        letter-spacing: -0.02em;
        margin: 0.5rem 0 0.3rem;
    }
    .section-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    .col-header {
        font-size: 0.72rem;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        margin: 0.5rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1f2937;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 1rem;
    }
    .metric-card {
        background: #141b2d;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1rem 0.75rem;
        text-align: center;
    }
    .metric-card.big { padding: 1.15rem; }
    .metric-label {
        font-size: 0.6rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.3rem;
        font-weight: 700;
        color: #00d4ff;
        margin-top: 0.35rem;
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    .metric-value.big { font-size: 1.65rem; }
    .metric-sub {
        font-size: 0.58rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 7px 18px;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .risk-badge.low { background: rgba(0, 230, 118, 0.12); color: #00e676; border: 1px solid rgba(0, 230, 118, 0.35); }
    .risk-badge.medium { background: rgba(255, 171, 0, 0.12); color: #ffab00; border: 1px solid rgba(255, 171, 0, 0.35); }
    .risk-badge.high { background: rgba(255, 82, 82, 0.12); color: #ff5252; border: 1px solid rgba(255, 82, 82, 0.35); }

    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #0099cc);
        color: #061019;
        border: none;
        border-radius: 10px;
        padding: 14px 20px;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
        letter-spacing: -0.01em;
        transition: all 0.2s;
        min-height: 52px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 14px 28px rgba(0, 212, 255, 0.35);
    }

    .empty-state {
        background: #141b2d;
        border: 1px dashed #2a3548;
        border-radius: 12px;
        padding: 3.5rem 2rem;
        text-align: center;
        color: #64748b;
    }
    .empty-state-icon { font-size: 3.2rem; margin-bottom: 1rem; opacity: 0.4; }
    .empty-state-text { font-size: 0.9rem; max-width: 320px; margin: 0 auto; line-height: 1.6; }

    .input-summary {
        background: #0a0e1a;
        border: 1px solid #1f2937;
        border-radius: 10px;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    .input-row {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px dashed #1f2937;
        font-size: 0.78rem;
    }
    .input-row:last-child { border-bottom: none; }
    .input-label { color: #94a3b8; font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; }
    .input-value { color: #00d4ff; font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.72rem; }

    /* Comparison diff bars */
    .diff-row {
        display: grid;
        grid-template-columns: 110px 1fr 110px;
        gap: 10px;
        align-items: center;
        padding: 6px 0;
    }
    .diff-label { font-size: 0.72rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace; }
    .diff-value { font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #141b2d;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1f2937;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 600;
        padding: 8px 16px;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 212, 255, 0.08) !important;
        color: #00d4ff !important;
    }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }

    @media (max-width: 768px) {
        .hero-title { font-size: 1.9rem; }
        .kpi-strip { grid-template-columns: repeat(2, 1fr); }
        .metric-grid { grid-template-columns: 1fr; }
        .diff-row { grid-template-columns: 80px 1fr 80px; }
    }
</style>
""")


# ============================================
# Cached loaders
# ============================================
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_metadata():
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_shap_explainer():
    return shap.TreeExplainer(load_model())


# ============================================
# Feature engineering
# ============================================
def compute_derived_features(d: dict) -> dict:
    safe = lambda x: 0 if x is None or (isinstance(x, float) and np.isnan(x)) else x
    loan_amnt = safe(d.get("loan_amnt", 0))
    annual_inc = safe(d.get("annual_inc", 0))
    revol_bal = safe(d.get("revol_bal", 0))
    open_acc = safe(d.get("open_acc", 0))
    total_acc = safe(d.get("total_acc", 0))
    delinq_2yrs = safe(d.get("delinq_2yrs", 0))
    pub_rec_bank = safe(d.get("pub_rec_bankruptcies", 0))
    mort_acc = safe(d.get("mort_acc", 0))
    fico_low = safe(d.get("fico_range_low", 0))
    fico_high = safe(d.get("fico_range_high", 0))
    installment = safe(d.get("installment", 0))
    monthly_income = annual_inc / 12
    return {
        "loan_income_ratio": loan_amnt / (annual_inc + 1),
        "monthly_income": monthly_income,
        "installment_to_income": installment / (monthly_income + 1),
        "balance_per_account": revol_bal / (open_acc + 1),
        "delinq_per_account": delinq_2yrs / (total_acc + 1),
        "log_annual_inc": float(np.log1p(annual_inc)),
        "log_revol_bal": float(np.log1p(revol_bal)),
        "log_loan_amnt": float(np.log1p(loan_amnt)),
        "fico_avg": (fico_low + fico_high) / 2,
        "fico_range": fico_high - fico_low,
        "total_credit_lines": open_acc + mort_acc,
        "has_delinq": 1 if delinq_2yrs > 0 else 0,
        "has_bankruptcy": 1 if pub_rec_bank > 0 else 0,
        "mths_since_last_delinq_missing": 0,
        "emp_length_missing": 0,
        "mort_acc_missing": 0,
    }


def build_feature_row(values: dict, feature_names: list) -> pd.DataFrame:
    merged = {**values, **compute_derived_features(values)}
    row = {k: merged.get(k, -1) for k in feature_names}
    df = pd.DataFrame([row])
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def build_feature_frame(rows: list, feature_names: list) -> pd.DataFrame:
    """Build a DataFrame for multiple rows (for batch prediction)."""
    records = []
    for r in rows:
        merged = {**r, **compute_derived_features(r)}
        records.append({k: merged.get(k, -1) for k in feature_names})
    df = pd.DataFrame(records)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ============================================
# Helpers
# ============================================
def risk_class(p: float) -> str:
    if p < 0.10: return "low"
    if p < 0.30: return "medium"
    return "high"


def risk_label(p: float) -> str:
    if p < 0.10: return "Low Risk"
    if p < 0.30: return "Moderate Risk"
    return "High Risk"


def pd_to_score(p: float) -> int:
    log_odds = np.log(p / (1 - p) + 0.001)
    score = 850 - 80 * (log_odds + 4)
    return int(round(max(300, min(850, score))))


def fmt_money(x: float) -> str:
    return f"${int(round(x)):,}"


def predict(model, values: dict, feature_names: list) -> float:
    return float(model.predict(build_feature_row(values, feature_names))[0])


def predict_batch(model, rows: list, feature_names: list) -> np.ndarray:
    df = build_feature_frame(rows, feature_names)
    return model.predict(df)


def compute_shap_top(explainer, values: dict, feature_names: list, top_n: int = 8):
    df = build_feature_row(values, feature_names)
    sv = explainer.shap_values(df)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv).reshape(-1)
    pairs = [{"feature": name, "impact": float(sv[i])} for i, name in enumerate(feature_names)]
    pairs.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return pairs[:top_n]


def build_grade_features(grade: str) -> dict:
    p = GRADE_PROFILES[grade]
    revol_bal = round((p["revol_util"] / 100) * p["annual_inc"] * 0.3)
    r = p["int_rate"] / 100 / 12
    installment = round(p["loan_amnt"] * r / (1 - (1 + r) ** (-p["term"])))
    return {
        "loan_amnt": p["loan_amnt"], "int_rate": p["int_rate"],
        "annual_inc": p["annual_inc"], "dti": p["dti"],
        "fico_range_low": p["fico"], "fico_range_high": p["fico"] + 4,
        "revol_util": p["revol_util"], "emp_length": p["emp_length"],
        "term": p["term"], "grade": grade, "sub_grade": f"{grade}3",
        "home_ownership": p["home_ownership"], "purpose": p["purpose"],
        "delinq_2yrs": 0, "inq_last_6mths": 0, "mths_since_last_delinq": 60,
        "open_acc": 10, "pub_rec": 0, "total_acc": 20,
        "application_type": "Individual", "mort_acc": 0,
        "pub_rec_bankruptcies": 0, "credit_history_months": 120,
        "verification_status": "Verified",
        "revol_bal": revol_bal, "installment": installment,
    }


@st.cache_data(show_spinner=False)
def compute_grade_stats(_model, feature_names_tuple: tuple) -> dict:
    feature_names = list(feature_names_tuple)
    stats = {}
    for grade in "ABCDEFG":
        features = build_grade_features(grade)
        pd_val = predict(_model, features, feature_names)
        stats[grade] = {
            "pd": pd_val, "int_rate": features["int_rate"],
            "loan_amnt": features["loan_amnt"],
            "el_per_loan": pd_val * LGD * features["loan_amnt"],
        }
    return stats


# ============================================
# Counterfactual
# ============================================
def find_counterfactual(model, values, feature_names, target_pd=0.10, max_iter=200):
    actionable = [
        ("fico_range_low", 610, 845, +1, 1.0), ("int_rate", 5.0, 30.0, -1, 0.9),
        ("dti", 0.0, 40.0, -1, 0.8), ("revol_util", 0, 120, -1, 0.7),
        ("annual_inc", 10000, 300000, +1, 0.5), ("loan_amnt", 500, 40000, -1, 0.4),
    ]
    current = {**values}
    current_pd = predict(model, current, feature_names)
    if current_pd <= target_pd:
        return {"achievable": True, "final_pd": current_pd, "changes": []}
    changes = []
    iterations = 0
    priority = sorted(actionable, key=lambda x: -x[4])
    for field, fmin, fmax, direction, weight in priority:
        if iterations >= max_iter:
            break
        current_val = current[field]
        best_val = current_val
        best_pd = predict(model, current, feature_names)
        for cand in np.linspace(fmin, fmax, 12):
            if direction > 0 and cand <= current_val: continue
            if direction < 0 and cand >= current_val: continue
            trial = {**current, field: float(cand)}
            if field == "fico_range_low": trial["fico_range_high"] = float(cand) + 4
            if field == "revol_util": trial["revol_bal"] = round((float(cand) / 100) * trial["annual_inc"] * 0.3)
            if field == "loan_amnt":
                _r = trial["int_rate"] / 100 / 12
                trial["installment"] = round(trial["loan_amnt"] * _r / (1 - (1 + _r) ** (-trial["term"])))
            trial_pd = predict(model, trial, feature_names)
            iterations += 1
            if trial_pd < best_pd:
                best_pd = trial_pd
                best_val = float(cand)
            if best_pd <= target_pd: break
        if best_val != current_val and best_pd < current_pd:
            old_pd = current_pd
            current[field] = best_val
            if field == "fico_range_low": current["fico_range_high"] = best_val + 4
            if field == "revol_util": current["revol_bal"] = round((best_val / 100) * current["annual_inc"] * 0.3)
            if field == "loan_amnt":
                _r = current["int_rate"] / 100 / 12
                current["installment"] = round(current["loan_amnt"] * _r / (1 - (1 + _r) ** (-current["term"])))
            changes.append({"field": field, "from": current_val, "to": best_val, "old_pd": old_pd, "new_pd": best_pd})
            current_pd = best_pd
        if current_pd <= target_pd: break
    if changes:
        total_drop = sum(c["old_pd"] - c["new_pd"] for c in changes)
        for c in changes:
            c["impact_share"] = (c["old_pd"] - c["new_pd"]) / total_drop if total_drop > 0 else 0
    return {"achievable": current_pd <= target_pd, "final_pd": current_pd, "changes": changes}


# ============================================
# Charts
# ============================================
def make_pd_gauge(pd_value: float) -> go.Figure:
    pct = pd_value * 100
    cls = risk_class(pd_value)
    color_map = {"low": COLOR_SUCCESS, "medium": COLOR_WARNING, "high": COLOR_DANGER}
    bar_color = color_map[cls]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number={"suffix": "%", "font": {"size": 46, "color": bar_color, "family": "JetBrains Mono"}},
        title={"text": "PROBABILITY OF DEFAULT", "font": {"size": 11, "color": COLOR_MUTED}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLOR_DIM,
                     "tickfont": {"size": 9, "color": COLOR_DIM}, "ticksuffix": "%"},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": COLOR_BG, "borderwidth": 0,
            "steps": [
                {"range": [0, 10], "color": "rgba(0, 230, 118, 0.08)"},
                {"range": [10, 30], "color": "rgba(255, 171, 0, 0.08)"},
                {"range": [30, 100], "color": "rgba(255, 82, 82, 0.08)"},
            ],
            "threshold": {"line": {"color": bar_color, "width": 4}, "thickness": 0.85, "value": pct},
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=45, b=10),
                       paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG)
    return fig


def make_shap_waterfall(shap_items: list) -> go.Figure:
    features = [x["feature"] for x in shap_items][::-1]
    impacts = [x["impact"] for x in shap_items][::-1]
    colors = [COLOR_DANGER if v > 0 else COLOR_SUCCESS for v in impacts]
    fig = go.Figure(go.Bar(
        x=impacts, y=features, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in impacts],
        textposition="outside",
        textfont=dict(size=11, color=COLOR_TEXT, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>Impact: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=max(220, 34 * len(features)),
        margin=dict(l=10, r=60, t=20, b=20),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(title="SHAP value", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, zerolinecolor=COLOR_ACCENT,
                   zerolinewidth=1, tickfont=dict(size=10, color=COLOR_DIM)),
        yaxis=dict(tickfont=dict(size=11, color=COLOR_TEXT, family="JetBrains Mono")),
        showlegend=False,
    )
    return fig


def make_sensitivity_chart(model, values, feature_names, vary_field, vary_range) -> go.Figure:
    pds = []
    base = {**values}
    for v in vary_range:
        base[vary_field] = float(v)
        if vary_field == "fico_range_low": base["fico_range_high"] = float(v) + 4
        if vary_field == "revol_util": base["revol_bal"] = round((float(v) / 100) * base["annual_inc"] * 0.3)
        pds.append(predict(model, base, feature_names) * 100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vary_range, y=pds, mode="lines",
        line=dict(color=COLOR_ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.12)",
        hovertemplate="%{x}<br>PD: %{y:.2f}%<extra></extra>"))
    current = values[vary_field]
    current_pd = predict(model, values, feature_names) * 100
    fig.add_trace(go.Scatter(x=[current], y=[current_pd], mode="markers",
        marker=dict(size=14, color=COLOR_DANGER, line=dict(color="white", width=2)),
        hovertemplate="Current: %{x}<br>PD: %{y:.2f}%<extra></extra>"))
    fig.update_layout(
        height=240, margin=dict(l=10, r=20, t=20, b=40),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(title=vary_field.replace("_", " ").title(),
                   title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        yaxis=dict(title="PD (%)", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        showlegend=False,
    )
    return fig


def make_portfolio_donut(distribution: dict) -> go.Figure:
    labels = list(distribution.keys()); values = list(distribution.values())
    colors = [GRADE_COLORS[g] for g in labels]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.6,
        marker=dict(colors=colors, line=dict(color=COLOR_BG, width=2)),
        textinfo="label+percent", textfont=dict(size=12, family="Inter", color=COLOR_TEXT),
        hovertemplate="<b>Grade %{label}</b><br>Loans: %{value}<br>%{percent}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG, showlegend=False)
    return fig


def make_portfolio_el_bar(portfolio: list) -> go.Figure:
    grades = [p["grade"] for p in portfolio]
    els = [p["el"] for p in portfolio]
    colors = [GRADE_COLORS[g] for g in grades]
    fig = go.Figure(go.Bar(x=grades, y=els, marker=dict(color=colors, line=dict(width=0)),
        text=[f"${e/1000:.0f}K" if e >= 1000 else f"${e:.0f}" for e in els],
        textposition="outside", textfont=dict(size=11, color=COLOR_TEXT, family="JetBrains Mono"),
        hovertemplate="<b>Grade %{x}</b><br>EL: $%{y:,.0f}<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(title="Grade", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=11, color=COLOR_TEXT)),
        yaxis=dict(title="Expected Loss ($)", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        showlegend=False)
    return fig


def make_portfolio_pd_bar(portfolio: list) -> go.Figure:
    grades = [p["grade"] for p in portfolio]
    pds = [p["pd"] * 100 for p in portfolio]
    colors = [GRADE_COLORS[g] for g in grades]
    fig = go.Figure(go.Bar(x=grades, y=pds, marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in pds], textposition="outside",
        textfont=dict(size=11, color=COLOR_TEXT, family="JetBrains Mono"),
        hovertemplate="<b>Grade %{x}</b><br>PD: %{y:.2f}%<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=30),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(title="Grade", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=11, color=COLOR_TEXT)),
        yaxis=dict(title="PD (%)", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        showlegend=False)
    return fig


def make_comparison_bars(a_vals: dict, b_vals: dict, fields: list) -> go.Figure:
    """Grouped bar chart for borrower comparison."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Borrower A", x=[f[0] for f in fields],
        y=[a_vals[f[1]] for f in fields],
        marker=dict(color=COLOR_ACCENT),
        text=[str(a_vals[f[1]]) for f in fields],
        textposition="outside",
        textfont=dict(size=10, color=COLOR_TEXT, family="JetBrains Mono"),
    ))
    fig.add_trace(go.Bar(
        name="Borrower B", x=[f[0] for f in fields],
        y=[b_vals[f[1]] for f in fields],
        marker=dict(color=COLOR_PURPLE),
        text=[str(b_vals[f[1]]) for f in fields],
        textposition="outside",
        textfont=dict(size=10, color=COLOR_TEXT, family="JetBrains Mono"),
    ))
    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        barmode="group", bargap=0.3, bargroupgap=0.1,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=COLOR_MUTED)),
        xaxis=dict(gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_TEXT)),
        yaxis=dict(gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
    )
    return fig


def make_batch_pd_distribution(pds: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pds * 100, nbinsx=30,
        marker=dict(color=COLOR_ACCENT, line=dict(color=COLOR_BG, width=1)),
        opacity=0.85,
        hovertemplate="PD: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(title="PD (%)", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        yaxis=dict(title="Count", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        showlegend=False, bargap=0.05,
    )
    return fig


def make_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_val: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
        line=dict(color=COLOR_ACCENT, width=3),
        fill="tozeroy", fillcolor="rgba(0, 212, 255, 0.1)",
        name=f"ROC (AUC = {auc_val:.4f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color=COLOR_DIM, width=2, dash="dash"),
        name="Random", hoverinfo="skip"))
    fig.update_layout(
        height=360, margin=dict(l=20, r=20, t=30, b=40),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(title="False Positive Rate", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM), range=[0, 1]),
        yaxis=dict(title="True Positive Rate", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM), range=[0, 1]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=COLOR_MUTED)),
    )
    return fig


def make_cost_curve(thresholds, total_costs, optimal_t) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=total_costs, mode="lines",
        line=dict(color=COLOR_DANGER, width=3),
        fill="tozeroy", fillcolor="rgba(255, 82, 82, 0.1)",
        hovertemplate="Threshold: %{x:.2f}<br>Cost: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=optimal_t, line=dict(color=COLOR_SUCCESS, width=2, dash="dash"),
                  annotation_text=f"Optimal: {optimal_t:.2f}",
                  annotation_position="top",
                  annotation_font=dict(color=COLOR_SUCCESS))
    fig.update_layout(
        height=320, margin=dict(l=20, r=20, t=30, b=40),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(title="Decision Threshold", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        yaxis=dict(title="Total Cost ($)", title_font=dict(size=11, color=COLOR_MUTED),
                   gridcolor=COLOR_BORDER, tickfont=dict(size=10, color=COLOR_DIM)),
        showlegend=False,
    )
    return fig


def make_confusion_heatmap(tp, fp, fn, tn) -> go.Figure:
    z = [[tn, fp], [fn, tp]]
    text = [[f"TN<br>{tn:,}", f"FP<br>{fp:,}"], [f"FN<br>{fn:,}", f"TP<br>{tp:,}"]]
    fig = go.Figure(go.Heatmap(
        z=z, text=text, texttemplate="%{text}",
        textfont=dict(size=14, family="JetBrains Mono", color=COLOR_TEXT),
        colorscale=[[0, "#141b2d"], [1, "#00d4ff"]],
        showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=COLOR_BG, plot_bgcolor=COLOR_BG,
        font={"family": "Inter, sans-serif", "size": 11, "color": COLOR_TEXT},
        xaxis=dict(tickvals=[0, 1], ticktext=["Predicted Neg", "Predicted Pos"],
                   tickfont=dict(size=11, color=COLOR_TEXT), side="bottom"),
        yaxis=dict(tickvals=[0, 1], ticktext=["Actual Neg", "Actual Pos"],
                   tickfont=dict(size=11, color=COLOR_TEXT), autorange="reversed"),
    )
    return fig


# ============================================
# Load resources
# ============================================
try:
    model = load_model()
    metadata = load_metadata()
    feature_names = model.feature_name()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()


# ============================================
# Session state defaults
# ============================================
for key, default in [
    ("analyzed", False), ("cf_result", None), ("cf_target_used", 10.0),
    ("comparison_ready", False), ("batch_ready", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================
# HERO + KPI
# ============================================
st.html("""
<div class="hero-wrap">
    <div class="hero-badge">Interactive · SHAP · Counterfactual · Portfolio · Batch · ROC</div>
    <h1 class="hero-title">Credit <span class="grad">Risk Suite</span></h1>
    <p class="hero-sub">
        The complete ML audit trail for credit risk.
        Predict, compare, simulate, batch-process, and analyze model behavior.
    </p>
</div>
""")

st.html(f"""
<div class="kpi-strip">
    <div class="kpi-item"><div class="kpi-label">AUC</div><div class="kpi-value">{AUC}</div></div>
    <div class="kpi-item"><div class="kpi-label">KS Statistic</div><div class="kpi-value">{KS}</div></div>
    <div class="kpi-item"><div class="kpi-label">Features</div><div class="kpi-value">{len(feature_names)}</div></div>
    <div class="kpi-item"><div class="kpi-label">Trees</div><div class="kpi-value">{model.num_trees()}</div></div>
</div>
""")


# ============================================
# TABS
# ============================================
tab_custom, tab_portfolio, tab_compare, tab_batch, tab_insights = st.tabs([
    "📊 Custom Analysis",
    "💼 Portfolio Simulator",
    "⚖️ Compare Borrowers",
    "📁 Batch Prediction",
    "🔬 Model Insights",
])


# ============================================================
# TAB 1: CUSTOM ANALYSIS
# ============================================================
with tab_custom:
    st.markdown("### Borrower Details")
    st.caption("Adjust values to match the borrower profile.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.html('<div class="col-header">💰 Loan Details</div>')
        loan_amnt = st.slider("Loan Amount ($)", 500, 40000, 15000, 500, format="$%d", key="f_loan_amnt")
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 14.5, 0.1, format="%.1f%%", key="f_int_rate")
        term = st.selectbox("Term", [36, 60], index=0, format_func=lambda x: f"{x} months", key="f_term")
        purpose = st.selectbox("Purpose",
            ["debt_consolidation", "credit_card", "home_improvement",
             "major_purchase", "small_business", "car", "medical"],
            index=0, format_func=lambda x: x.replace("_", " ").title(), key="f_purpose")

    with col2:
        st.html('<div class="col-header">👤 Borrower Profile</div>')
        annual_inc = st.slider("Annual Income ($)", 10000, 300000, 60000, 1000, format="$%d", key="f_annual_inc")
        emp_length = st.slider("Employment Length (yrs)", 0, 10, 4, 1, key="f_emp_length")
        home_ownership = st.selectbox("Home Ownership", ["MORTGAGE", "OWN", "RENT"], index=2,
            format_func=lambda x: {"MORTGAGE": "Mortgage", "OWN": "Own", "RENT": "Rent"}[x],
            key="f_home_ownership")
        dti = st.slider("Debt-to-Income (%)", 0.0, 40.0, 20.0, 0.5, format="%.1f%%", key="f_dti")

    with col3:
        st.html('<div class="col-header">📊 Credit Profile</div>')
        fico_range_low = st.slider("FICO Score", 610, 845, 690, 1, key="f_fico")
        revol_util = st.slider("Revolving Utilization (%)", 0, 120, 55, 1, format="%d%%", key="f_revol_util")
        grade = st.selectbox("Grade", list("ABCDEFG"), index=2, format_func=lambda x: f"Grade {x}", key="f_grade")
        sub_grades = [f"{grade}{i}" for i in range(1, 6)]
        sub_grade = st.selectbox("Sub-grade", sub_grades, index=2, key="f_sub_grade")

    st.write("")
    analyze_clicked = st.button("⚡ Analyze Risk", key="analyze_btn", use_container_width=True)

    current_values = {
        "loan_amnt": loan_amnt, "int_rate": int_rate, "annual_inc": annual_inc,
        "dti": dti, "fico_range_low": fico_range_low, "fico_range_high": fico_range_low + 4,
        "revol_util": revol_util, "emp_length": emp_length, "term": term,
        "grade": grade, "sub_grade": sub_grade, "home_ownership": home_ownership,
        "purpose": purpose, "delinq_2yrs": 0, "inq_last_6mths": 0,
        "mths_since_last_delinq": 60, "open_acc": 10, "pub_rec": 0, "total_acc": 20,
        "application_type": "Individual",
        "mort_acc": 2 if home_ownership == "MORTGAGE" else 0,
        "pub_rec_bankruptcies": 0, "credit_history_months": 120,
        "verification_status": "Verified",
    }
    current_values["revol_bal"] = round((revol_util / 100) * annual_inc * 0.3)
    _r = int_rate / 100 / 12
    current_values["installment"] = round(loan_amnt * _r / (1 - (1 + _r) ** (-term)))

    if analyze_clicked:
        with st.spinner("Computing prediction + SHAP..."):
            _pd = predict(model, current_values, feature_names)
            _explainer = load_shap_explainer()
            _shap = compute_shap_top(_explainer, current_values, feature_names, top_n=8)
        st.session_state["analyzed"] = True
        st.session_state["analyzed_values"] = current_values.copy()
        st.session_state["analyzed_pd"] = _pd
        st.session_state["analyzed_shap"] = _shap
        st.session_state["cf_result"] = None

    st.write("")

    if not st.session_state.get("analyzed", False):
        st.html("""
<div class="empty-state">
    <div class="empty-state-icon">📊</div>
    <div class="empty-state-text">
        Adjust the borrower details above, then click
        <strong style="color: #00d4ff;">Analyze Risk</strong>
        to see the prediction with SHAP explanation.
    </div>
</div>
""")
    else:
        av = st.session_state["analyzed_values"]
        pd_value = st.session_state["analyzed_pd"]
        shap_items = st.session_state["analyzed_shap"]
        credit_score = pd_to_score(pd_value)
        el = pd_value * LGD * av["loan_amnt"]
        min_rate = (COST_OF_FUNDS + pd_value * LGD + OPERATING_COST + PROFIT_MARGIN) * 100
        cls = risk_class(pd_value)

        col_reset, _ = st.columns([1, 4])
        with col_reset:
            if st.button("🔄 Start Over", key="reset_btn"):
                st.session_state["analyzed"] = False
                st.session_state["cf_result"] = None
                st.rerun()

        st.html('<div class="section-title">Prediction Result</div>')

        col_a, col_b = st.columns([1, 1.2])
        with col_a:
            st.plotly_chart(make_pd_gauge(pd_value), use_container_width=True, config={"displayModeBar": False})
            st.html(
                f'<div style="text-align: center; margin-top: -10px;">'
                f'<span class="risk-badge {cls}">{risk_label(pd_value)}</span></div>')
        with col_b:
            st.html(f"""
<div class="metric-grid">
    <div class="metric-card"><div class="metric-label">Credit Score</div>
        <div class="metric-value">{credit_score}</div>
        <div class="metric-sub">300 – 850 scale</div></div>
    <div class="metric-card"><div class="metric-label">Expected Loss</div>
        <div class="metric-value">{fmt_money(el)}</div>
        <div class="metric-sub">PD × LGD × EAD</div></div>
    <div class="metric-card"><div class="metric-label">Suggested Rate</div>
        <div class="metric-value">{min_rate:.2f}%</div>
        <div class="metric-sub">Risk-based pricing</div></div>
    <div class="metric-card"><div class="metric-label">Loan Amount</div>
        <div class="metric-value">{fmt_money(av['loan_amnt'])}</div>
        <div class="metric-sub">{av['term']} months</div></div>
</div>
""")

        # Export buttons
        st.write("")
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            # Generate HTML report for download
            report_html = f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CreditTrace Report</title>
<style>body{{font-family:Arial;background:#0a0e1a;color:#e8eef5;padding:40px;}}
h1{{color:#00d4ff;}} .box{{background:#141b2d;padding:20px;border-radius:10px;margin:15px 0;border:1px solid #1f2937;}}
.k{{color:#94a3b8;font-size:13px;}} .v{{font-size:22px;font-weight:bold;color:#00d4ff;font-family:monospace;}}
table{{width:100%;}}td{{padding:6px 0;}}</style></head>
<body>
<h1>CreditTrace — Risk Report</h1>
<p style="color:#64748b;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<div class="box"><div class="k">PROBABILITY OF DEFAULT</div>
<div class="v">{pd_value*100:.2f}% — {risk_label(pd_value)}</div></div>
<div class="box"><table>
<tr><td class="k">Credit Score</td><td class="v">{credit_score}</td></tr>
<tr><td class="k">Expected Loss</td><td class="v">{fmt_money(el)}</td></tr>
<tr><td class="k">Suggested Rate</td><td class="v">{min_rate:.2f}%</td></tr>
<tr><td class="k">Loan Amount</td><td class="v">{fmt_money(av['loan_amnt'])}</td></tr>
<tr><td class="k">Term</td><td class="v">{av['term']} months</td></tr>
</table></div>
<div class="box"><div class="k">INPUT PARAMETERS</div><table>
<tr><td class="k">Interest Rate</td><td>{av['int_rate']:.2f}%</td></tr>
<tr><td class="k">Annual Income</td><td>{fmt_money(av['annual_inc'])}</td></tr>
<tr><td class="k">DTI</td><td>{av['dti']:.1f}%</td></tr>
<tr><td class="k">FICO</td><td>{av['fico_range_low']}</td></tr>
<tr><td class="k">Grade</td><td>{av['grade']} ({av['sub_grade']})</td></tr>
<tr><td class="k">Home Ownership</td><td>{av['home_ownership']}</td></tr>
<tr><td class="k">Purpose</td><td>{av['purpose']}</td></tr>
<tr><td class="k">Revolving Utilization</td><td>{av['revol_util']}%</td></tr>
</table></div>
<div class="box"><div class="k">TOP SHAP CONTRIBUTORS</div><table>
{''.join(f'<tr><td class="k">{i["feature"]}</td><td>{i["impact"]:+.4f}</td></tr>' for i in shap_items[:8])}
</table></div>
<p style="color:#64748b;font-size:12px;text-align:center;margin-top:40px;">
CreditTrace — Credit Risk, Engineered.</p>
</body></html>
"""
            st.download_button(
                "📄 Download HTML Report",
                data=report_html,
                file_name=f"credittrace_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                key="dl_report",
                use_container_width=True,
            )
        with col_exp2:
            # Generate CSV for download
            csv_rows = [{"field": k, "value": str(v)} for k, v in av.items()]
            csv_rows.append({"field": "predicted_pd", "value": f"{pd_value:.6f}"})
            csv_rows.append({"field": "credit_score", "value": str(credit_score)})
            csv_rows.append({"field": "expected_loss", "value": f"{el:.2f}"})
            csv_rows.append({"field": "suggested_rate_pct", "value": f"{min_rate:.2f}"})
            csv_data = pd.DataFrame(csv_rows).to_csv(index=False)
            st.download_button(
                "📊 Download CSV",
                data=csv_data,
                file_name=f"credittrace_prediction_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="dl_csv",
                use_container_width=True,
            )

        st.write("")
        st.html('<div class="section-title">Why this prediction?</div>')
        st.html('<div class="section-sub">SHAP values show how each feature pushed the prediction.</div>')

        col_w, col_i = st.columns([1.4, 1])
        with col_w:
            st.plotly_chart(make_shap_waterfall(shap_items), use_container_width=True,
                            config={"displayModeBar": False})
        with col_i:
            st.markdown("**Top contributors**")
            for item in shap_items[:5]:
                direction = "↑ toward default" if item["impact"] > 0 else "↓ away from default"
                color = COLOR_DANGER if item["impact"] > 0 else COLOR_SUCCESS
                st.html(
                    '<div style="padding: 6px 0; border-bottom: 1px solid #1f2937;">'
                    f'<div style="font-family: JetBrains Mono; font-size: 0.78rem;">{item["feature"]}</div>'
                    f'<div style="font-size: 0.7rem; color: {color}; margin-top: 2px;">'
                    f'{item["impact"]:+.4f} · {direction}</div></div>')

        st.write("")
        st.html('<div class="section-title">Sensitivity Analysis</div>')
        st.html('<div class="section-sub">How PD changes as you vary one feature.</div>')

        sens_options = {
            "Interest Rate (%)": ("int_rate", np.linspace(5, 30, 30)),
            "FICO Score": ("fico_range_low", np.linspace(610, 845, 30)),
            "DTI (%)": ("dti", np.linspace(0, 40, 30)),
            "Revolving Utilization (%)": ("revol_util", np.linspace(0, 120, 30)),
            "Loan Amount ($)": ("loan_amnt", np.linspace(500, 40000, 30)),
            "Annual Income ($)": ("annual_inc", np.linspace(10000, 300000, 30)),
        }
        sens_choice = st.selectbox("Vary which feature?", list(sens_options.keys()),
                                    index=0, key="sens_choice")
        vary_field, vary_range = sens_options[sens_choice]
        st.plotly_chart(make_sensitivity_chart(model, av, feature_names, vary_field, vary_range),
                        use_container_width=True, config={"displayModeBar": False})

        st.write("")
        st.html('<div class="section-title">What would improve this profile?</div>')
        st.html('<div class="section-sub">Minimum changes needed to reach a target PD.</div>')

        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            target_pct = st.slider("Target PD (%)", 3.0, 25.0, 10.0, 0.5, format="%.1f%%", key="cf_target")
        with col_ctrl2:
            st.write(""); st.write("")
            run_cf = st.button("🎯 Find Minimum Changes", key="cf_btn")

        if run_cf:
            with st.spinner("Searching for the best combination..."):
                cf = find_counterfactual(model, av, feature_names, target_pd=target_pct / 100.0)
            st.session_state["cf_result"] = cf
            st.session_state["cf_target_used"] = target_pct

        if st.session_state.get("cf_result") is not None:
            cf = st.session_state["cf_result"]
            target_used = st.session_state.get("cf_target_used", 10.0)
            if cf["achievable"]:
                st.success(f"✅ Achievable: PD can drop from **{pd_value*100:.2f}%** "
                           f"to **{cf['final_pd']*100:.2f}%** (target: {target_used:.1f}%)")
            else:
                st.warning(f"⚠️ Target not fully reachable. Best achieved: "
                           f"**{cf['final_pd']*100:.2f}%**")
            if cf["changes"]:
                st.markdown("**Recommended changes:**")
                def _fmt(v, field):
                    if field in ("annual_inc", "loan_amnt"): return fmt_money(v)
                    if field in ("int_rate", "dti"): return f"{v:.1f}%"
                    if field == "revol_util": return f"{v:.0f}%"
                    return f"{v:.0f}"
                for change in cf["changes"]:
                    field_label = {
                        "fico_range_low": "FICO Score", "int_rate": "Interest Rate",
                        "dti": "Debt-to-Income", "revol_util": "Revolving Utilization",
                        "annual_inc": "Annual Income", "loan_amnt": "Loan Amount",
                    }.get(change["field"], change["field"])
                    from_str = _fmt(change["from"], change["field"])
                    to_str = _fmt(change["to"], change["field"])
                    delta = change["to"] - change["from"]
                    if change["field"] in ("annual_inc", "loan_amnt"):
                        delta_str = f"{int(delta):+,d}"
                    elif change["field"] == "fico_range_low":
                        delta_str = f"{int(delta):+d} pts"
                    else:
                        delta_str = f"{delta:+.1f}"
                    impact_pct = change["impact_share"] * 100
                    st.html(
                        '<div style="background: #141b2d; border: 1px solid #1f2937;'
                        ' border-radius: 10px; padding: 1rem; margin-bottom: 0.75rem;">'
                        '<div style="display: flex; justify-content: space-between;'
                        ' align-items: baseline; margin-bottom: 0.5rem; flex-wrap: wrap; gap: 8px;">'
                        f'<span style="font-weight: 600; color: #e8eef5;">{field_label}</span>'
                        '<span style="font-family: JetBrains Mono; color: #00d4ff;'
                        f' font-size: 0.82rem; font-weight: 600;">'
                        f'{from_str} → {to_str}'
                        f'<span style="color: #94a3b8; margin-left: 6px;">({delta_str})</span></span></div>'
                        '<div style="background: #0a0e1a; border-radius: 4px; height: 6px; overflow: hidden;">'
                        f'<div style="width: {impact_pct}%; height: 100%;'
                        ' background: linear-gradient(90deg, #00d4ff, #8b5cf6); border-radius: 4px;"></div></div>'
                        '<div style="font-size: 0.68rem; color: #94a3b8; margin-top: 0.4rem;'
                        f' font-family: JetBrains Mono;">{impact_pct:.0f}% of total improvement</div></div>')

        st.write("")
        with st.expander("📋 Input Parameters Used", expanded=False):
            rows = [
                ("Loan Amount", fmt_money(av["loan_amnt"])),
                ("Interest Rate", f"{av['int_rate']:.2f}%"),
                ("Term", f"{av['term']} months"),
                ("Purpose", av["purpose"].replace("_", " ").title()),
                ("Annual Income", fmt_money(av["annual_inc"])),
                ("Employment Length", f"{av['emp_length']} years"),
                ("Home Ownership", av["home_ownership"].title()),
                ("Debt-to-Income", f"{av['dti']:.1f}%"),
                ("FICO Score", str(av["fico_range_low"])),
                ("Revolving Utilization", f"{av['revol_util']}%"),
                ("Grade", av["grade"]),
                ("Sub-grade", av["sub_grade"]),
            ]
            rows_html = "".join(
                '<div class="input-row">'
                f'<span class="input-label">{k}</span>'
                f'<span class="input-value">{v}</span></div>'
                for k, v in rows)
            st.html(f'<div class="input-summary">{rows_html}</div>')


# ============================================================
# TAB 2: PORTFOLIO SIMULATOR
# ============================================================
with tab_portfolio:
    st.html('<div class="section-title">Portfolio Simulator</div>')
    st.html('<div class="section-sub">Build a synthetic loan portfolio by setting the grade distribution.</div>')

    grade_stats = compute_grade_stats(model, tuple(feature_names))

    col_ctrl, col_res = st.columns([1, 1.6])

    with col_ctrl:
        st.markdown("#### Portfolio Size")
        total_loans = st.slider("Number of loans", 100, 5000, 1000, 100, key="pf_loans")
        st.markdown("#### Grade Distribution")
        st.caption("Adjust the weights — they will be normalized to 100%.")
        weights = {}
        for g in "ABCDEFG":
            weights[g] = st.slider(f"Grade {g}", 0, 100, DEFAULT_WEIGHTS[g], 1, key=f"pf_w_{g}")
        total_w = sum(weights.values())
        st.markdown("#### Reset")
        def _reset_weights():
            for g in "ABCDEFG":
                st.session_state[f"pf_w_{g}"] = DEFAULT_WEIGHTS[g]
        st.button("↺ Reset to Default", key="pf_reset", on_click=_reset_weights)

    with col_res:
        if total_w == 0:
            st.error("Please give at least one grade a weight > 0.")
        else:
            normalized = {g: weights[g] / total_w for g in "ABCDEFG"}
            portfolio = []
            total_el = 0.0
            total_exposure = 0.0
            total_weighted_pd = 0.0
            for g in "ABCDEFG":
                count = int(round(normalized[g] * total_loans))
                if count == 0: continue
                stats = grade_stats[g]
                el_val = count * stats["el_per_loan"]
                exposure = count * stats["loan_amnt"]
                portfolio.append({"grade": g, "count": count, "pd": stats["pd"],
                    "el": el_val, "exposure": exposure,
                    "avg_loan": stats["loan_amnt"], "rate": stats["int_rate"]})
                total_el += el_val
                total_exposure += exposure
                total_weighted_pd += count * stats["pd"]
            avg_pd = total_weighted_pd / total_loans if total_loans > 0 else 0
            el_rate = total_el / total_exposure if total_exposure > 0 else 0

            st.html(f"""
<div class="metric-grid" style="grid-template-columns: 1fr 1fr;">
    <div class="metric-card big"><div class="metric-label">Total Expected Loss</div>
        <div class="metric-value big" style="color: #ff5252;">{fmt_money(total_el)}</div>
        <div class="metric-sub">Sum over all loans</div></div>
    <div class="metric-card big"><div class="metric-label">Total Exposure</div>
        <div class="metric-value big">{fmt_money(total_exposure)}</div>
        <div class="metric-sub">Sum of loan amounts</div></div>
</div>
<div class="metric-grid" style="grid-template-columns: 1fr 1fr; margin-top: 10px;">
    <div class="metric-card"><div class="metric-label">Portfolio PD</div>
        <div class="metric-value">{avg_pd*100:.2f}%</div>
        <div class="metric-sub">Weighted by loan count</div></div>
    <div class="metric-card"><div class="metric-label">EL Rate</div>
        <div class="metric-value">{el_rate*100:.2f}%</div>
        <div class="metric-sub">EL / Exposure</div></div>
</div>
""")
            st.write("")
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("**Composition**")
                donut_data = {p["grade"]: p["count"] for p in portfolio}
                st.plotly_chart(make_portfolio_donut(donut_data),
                                use_container_width=True, config={"displayModeBar": False})
            with col_chart2:
                st.markdown("**Expected Loss by Grade**")
                st.plotly_chart(make_portfolio_el_bar(portfolio),
                                use_container_width=True, config={"displayModeBar": False})
            st.markdown("**Probability of Default by Grade**")
            st.plotly_chart(make_portfolio_pd_bar(portfolio),
                            use_container_width=True, config={"displayModeBar": False})

            st.markdown("**Portfolio Breakdown**")
            table_rows = []
            for p in portfolio:
                share = p["el"] / total_el * 100 if total_el > 0 else 0
                table_rows.append(
                    '<div class="input-row">'
                    f'<span class="input-label" style="font-weight: 600;">Grade {p["grade"]} · {p["count"]} loans</span>'
                    f'<span class="input-value">{fmt_money(p["el"])} ({share:.0f}%)</span>'
                    '</div>')
            st.html('<div class="input-summary">' + "".join(table_rows) + '</div>')

            st.write("")
            st.markdown("**💡 Interpretation**")
            high_risk_share = sum(p["el"] for p in portfolio if p["grade"] in "EFG") / total_el * 100 if total_el > 0 else 0
            if high_risk_share > 60:
                st.warning(f"⚠️ High-risk grades (E, F, G) contribute **{high_risk_share:.0f}%** of total EL. Very risky portfolio.")
            elif high_risk_share > 30:
                st.info(f"📊 High-risk grades contribute **{high_risk_share:.0f}%**. Consider balancing.")
            else:
                st.success(f"✅ Conservative portfolio: high-risk grades contribute only **{high_risk_share:.0f}%**.")


# ============================================================
# TAB 3: COMPARE BORROWERS
# ============================================================
with tab_compare:
    st.html('<div class="section-title">Compare Two Borrowers</div>')
    st.html('<div class="section-sub">Side-by-side analysis with visual diffs. Understand what drives different risk levels.</div>')

    col_a, col_b = st.columns(2)

    with col_a:
        st.html('<div class="col-header" style="color: #00d4ff;">🟢 Borrower A</div>')
        a_grade = st.selectbox("Grade", list("ABCDEFG"), index=0, key="ca_grade", format_func=lambda x: f"Grade {x}")
        a_fico = st.slider("FICO Score", 610, 845, 810, 1, key="ca_fico")
        a_int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 7.5, 0.1, key="ca_rate")
        a_dti = st.slider("DTI (%)", 0.0, 40.0, 10.0, 0.5, key="ca_dti")
        a_loan_amnt = st.slider("Loan Amount ($)", 500, 40000, 15000, 500, key="ca_loan")
        a_annual_inc = st.slider("Annual Income ($)", 10000, 300000, 110000, 1000, key="ca_income")
        a_revol_util = st.slider("Revolving Util (%)", 0, 120, 20, 1, key="ca_revol")
        a_term = st.selectbox("Term", [36, 60], index=0, key="ca_term")

    with col_b:
        st.html('<div class="col-header" style="color: #8b5cf6;">🔴 Borrower B</div>')
        b_grade = st.selectbox("Grade", list("ABCDEFG"), index=4, key="cb_grade", format_func=lambda x: f"Grade {x}")
        b_fico = st.slider("FICO Score", 610, 845, 655, 1, key="cb_fico")
        b_int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 21.0, 0.1, key="cb_rate")
        b_dti = st.slider("DTI (%)", 0.0, 40.0, 28.0, 0.5, key="cb_dti")
        b_loan_amnt = st.slider("Loan Amount ($)", 500, 40000, 18000, 500, key="cb_loan")
        b_annual_inc = st.slider("Annual Income ($)", 10000, 300000, 55000, 1000, key="cb_income")
        b_revol_util = st.slider("Revolving Util (%)", 0, 120, 70, 1, key="cb_revol")
        b_term = st.selectbox("Term", [36, 60], index=1, key="cb_term")

    def _build_borrower(grade, fico, int_rate, dti, loan_amnt, annual_inc, revol_util, term):
        d = {
            "loan_amnt": loan_amnt, "int_rate": int_rate, "annual_inc": annual_inc,
            "dti": dti, "fico_range_low": fico, "fico_range_high": fico + 4,
            "revol_util": revol_util, "emp_length": 5, "term": term,
            "grade": grade, "sub_grade": f"{grade}3",
            "home_ownership": "RENT", "purpose": "debt_consolidation",
            "delinq_2yrs": 0, "inq_last_6mths": 0, "mths_since_last_delinq": 60,
            "open_acc": 10, "pub_rec": 0, "total_acc": 20,
            "application_type": "Individual", "mort_acc": 0,
            "pub_rec_bankruptcies": 0, "credit_history_months": 120,
            "verification_status": "Verified",
        }
        d["revol_bal"] = round((revol_util / 100) * annual_inc * 0.3)
        r = int_rate / 100 / 12
        d["installment"] = round(loan_amnt * r / (1 - (1 + r) ** (-term)))
        return d

    st.write("")
    run_compare = st.button("⚖️ Compare Borrowers", key="compare_btn", use_container_width=True)

    if run_compare:
        a_vals = _build_borrower(a_grade, a_fico, a_int_rate, a_dti, a_loan_amnt, a_annual_inc, a_revol_util, a_term)
        b_vals = _build_borrower(b_grade, b_fico, b_int_rate, b_dti, b_loan_amnt, b_annual_inc, b_revol_util, b_term)
        with st.spinner("Computing predictions..."):
            a_pd = predict(model, a_vals, feature_names)
            b_pd = predict(model, b_vals, feature_names)
            explainer = load_shap_explainer()
            a_shap = compute_shap_top(explainer, a_vals, feature_names, top_n=5)
            b_shap = compute_shap_top(explainer, b_vals, feature_names, top_n=5)
        st.session_state["comparison_data"] = {
            "a_vals": a_vals, "b_vals": b_vals,
            "a_pd": a_pd, "b_pd": b_pd,
            "a_shap": a_shap, "b_shap": b_shap,
        }
        st.session_state["comparison_ready"] = True

    if st.session_state.get("comparison_ready", False):
        data = st.session_state["comparison_data"]
        a_vals, b_vals = data["a_vals"], data["b_vals"]
        a_pd, b_pd = data["a_pd"], data["b_pd"]

        st.write("")
        # Headline comparison
        col_head1, col_head2 = st.columns(2)
        with col_head1:
            cls_a = risk_class(a_pd)
            st.html(f"""
<div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.08), transparent); border: 1px solid {COLOR_ACCENT}; border-radius: 12px; padding: 1.5rem; text-align: center;">
    <div style="font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Borrower A</div>
    <div style="font-family: JetBrains Mono; font-size: 3rem; font-weight: 700; color: {COLOR_ACCENT}; line-height: 1; margin: 0.5rem 0;">{a_pd*100:.2f}%</div>
    <span class="risk-badge {cls_a}">{risk_label(a_pd)}</span>
    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.75rem;">Credit Score: {pd_to_score(a_pd)}</div>
</div>
""")
        with col_head2:
            cls_b = risk_class(b_pd)
            st.html(f"""
<div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.08), transparent); border: 1px solid {COLOR_PURPLE}; border-radius: 12px; padding: 1.5rem; text-align: center;">
    <div style="font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;">Borrower B</div>
    <div style="font-family: JetBrains Mono; font-size: 3rem; font-weight: 700; color: {COLOR_PURPLE}; line-height: 1; margin: 0.5rem 0;">{b_pd*100:.2f}%</div>
    <span class="risk-badge {cls_b}">{risk_label(b_pd)}</span>
    <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 0.75rem;">Credit Score: {pd_to_score(b_pd)}</div>
</div>
""")

        # Delta
        st.write("")
        delta_pd = (b_pd - a_pd) * 100
        delta_el = (b_pd - a_pd) * LGD * b_vals["loan_amnt"]
        sign = "+" if delta_pd > 0 else ""
        st.html(f"""
<div style="background: #141b2d; border: 1px solid #1f2937; border-radius: 12px; padding: 1rem; text-align: center;">
    <div style="font-size: 0.75rem; color: #94a3b8;">Borrower B minus Borrower A</div>
    <div style="font-family: JetBrains Mono; font-size: 1.5rem; font-weight: 700; color: {COLOR_DANGER if delta_pd > 0 else COLOR_SUCCESS}; margin-top: 0.4rem;">
        {sign}{delta_pd:.2f}% PD difference
    </div>
</div>
""")

        # Grouped comparison chart
        st.write("")
        st.markdown("**Feature Comparison**")
        fields = [
            ("FICO", "fico_range_low"),
            ("Interest %", "int_rate"),
            ("DTI %", "dti"),
            ("Revol %", "revol_util"),
            ("Income ($K)", "annual_inc"),
            ("Loan ($K)", "loan_amnt"),
        ]
        # Normalize for chart
        a_chart = dict(a_vals); b_chart = dict(b_vals)
        for _, key in fields:
            if key in ("annual_inc", "loan_amnt"):
                a_chart[key] = a_vals[key] / 1000
                b_chart[key] = b_vals[key] / 1000
        st.plotly_chart(make_comparison_bars(a_chart, b_chart, fields),
                        use_container_width=True, config={"displayModeBar": False})

        # Diff bars
        st.markdown("**Key Differences**")
        diffs = [
            ("FICO Score", a_vals["fico_range_low"], b_vals["fico_range_low"], ""),
            ("Interest Rate", a_vals["int_rate"], b_vals["int_rate"], "%"),
            ("DTI", a_vals["dti"], b_vals["dti"], "%"),
            ("Revolving Util", a_vals["revol_util"], b_vals["revol_util"], "%"),
            ("Loan Amount", a_vals["loan_amnt"], b_vals["loan_amnt"], "$"),
        ]
        for label, va, vb, suffix in diffs:
            delta = vb - va
            if suffix == "$":
                va_str = fmt_money(va); vb_str = fmt_money(vb)
                delta_str = f"{'+' if delta > 0 else ''}{fmt_money(delta)}"
            else:
                va_str = f"{va:.0f}{suffix}"; vb_str = f"{vb:.0f}{suffix}"
                delta_str = f"{delta:+.0f}{suffix}"
            max_v = max(abs(va), abs(vb), 1)
            a_pct = (va / max_v) * 100
            b_pct = (vb / max_v) * 100
            st.html(f"""
<div class="diff-row">
    <div class="diff-value" style="color: {COLOR_ACCENT}; text-align: right;">{va_str}</div>
    <div>
        <div style="font-size: 0.65rem; color: #94a3b8; text-align: center; margin-bottom: 3px;">{label} <span style="color: {COLOR_DANGER if delta > 0 else COLOR_SUCCESS}; font-family: JetBrains Mono;">({delta_str})</span></div>
        <div style="background: #0a0e1a; border-radius: 4px; height: 6px; overflow: hidden; margin-bottom: 2px;">
            <div style="width: {a_pct}%; height: 100%; background: {COLOR_ACCENT}; border-radius: 4px;"></div>
        </div>
        <div style="background: #0a0e1a; border-radius: 4px; height: 6px; overflow: hidden;">
            <div style="width: {b_pct}%; height: 100%; background: {COLOR_PURPLE}; border-radius: 4px;"></div>
        </div>
    </div>
    <div class="diff-value" style="color: {COLOR_PURPLE};">{vb_str}</div>
</div>
""")

        # SHAP side by side
        st.write("")
        st.markdown("**SHAP Comparison**")
        col_shap1, col_shap2 = st.columns(2)
        with col_shap1:
            st.markdown("**Borrower A**")
            st.plotly_chart(make_shap_waterfall(data["a_shap"]),
                            use_container_width=True, config={"displayModeBar": False})
        with col_shap2:
            st.markdown("**Borrower B**")
            st.plotly_chart(make_shap_waterfall(data["b_shap"]),
                            use_container_width=True, config={"displayModeBar": False})
    else:
        st.html("""
<div class="empty-state">
    <div class="empty-state-icon">⚖️</div>
    <div class="empty-state-text">
        Adjust both profiles above, then click
        <strong style="color: #00d4ff;">Compare Borrowers</strong>
        to see a side-by-side analysis.
    </div>
</div>
""")


# ============================================================
# TAB 4: BATCH PREDICTION
# ============================================================
with tab_batch:
    st.html('<div class="section-title">Batch Prediction</div>')
    st.html('<div class="section-sub">Upload a CSV with borrower data — get predictions for all rows at once.</div>')

    st.markdown("**1. Download the template**")
    template_rows = []
    for g in "ABCDEFG":
        p = GRADE_PROFILES[g]
        template_rows.append({
            "loan_amnt": p["loan_amnt"], "term": p["term"], "int_rate": p["int_rate"],
            "grade": g, "sub_grade": f"{g}3", "emp_length": p["emp_length"],
            "home_ownership": p["home_ownership"], "annual_inc": p["annual_inc"],
            "verification_status": "Verified", "purpose": p["purpose"],
            "dti": p["dti"], "delinq_2yrs": 0, "fico_range_low": p["fico"],
            "fico_range_high": p["fico"] + 4, "inq_last_6mths": 0,
            "mths_since_last_delinq": 60, "open_acc": 10, "pub_rec": 0,
            "revol_bal": round((p["revol_util"] / 100) * p["annual_inc"] * 0.3),
            "revol_util": p["revol_util"], "total_acc": 20,
            "application_type": "Individual", "mort_acc": 0,
            "pub_rec_bankruptcies": 0, "credit_history_months": 120,
        })
    template_df = pd.DataFrame(template_rows)
    template_csv = template_df.to_csv(index=False)

    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.download_button(
            "📥 Download Template CSV",
            data=template_csv,
            file_name="credittrace_batch_template.csv",
            mime="text/csv",
            key="dl_template",
            use_container_width=True,
        )

    st.markdown("**2. Upload your CSV**")
    uploaded_file = st.file_uploader(
        "Choose a CSV file (or use the template above)",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(batch_df)} rows, {len(batch_df.columns)} columns")

            with st.expander("Preview uploaded data"):
                st.dataframe(batch_df.head(10), use_container_width=True)

            # Required cols
            required_cols = [
                "loan_amnt", "term", "int_rate", "grade", "sub_grade",
                "annual_inc", "dti", "fico_range_low",
            ]
            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            else:
                run_batch = st.button("🚀 Run Batch Prediction", key="run_batch", use_container_width=True)

                if run_batch:
                    with st.spinner(f"Predicting {len(batch_df)} rows..."):
                        # Build rows
                        rows_for_model = []
                        for _, row in batch_df.iterrows():
                            d = row.to_dict()
                            # Fill defaults for missing fields
                            d.setdefault("emp_length", 5)
                            d.setdefault("home_ownership", "RENT")
                            d.setdefault("verification_status", "Verified")
                            d.setdefault("purpose", "debt_consolidation")
                            d.setdefault("delinq_2yrs", 0)
                            d.setdefault("inq_last_6mths", 0)
                            d.setdefault("mths_since_last_delinq", 60)
                            d.setdefault("open_acc", 10)
                            d.setdefault("pub_rec", 0)
                            d.setdefault("total_acc", 20)
                            d.setdefault("application_type", "Individual")
                            d.setdefault("mort_acc", 0)
                            d.setdefault("pub_rec_bankruptcies", 0)
                            d.setdefault("credit_history_months", 120)
                            d["fico_range_high"] = d.get("fico_range_high", d["fico_range_low"] + 4)
                            d["revol_bal"] = d.get("revol_bal", round((d.get("revol_util", 50) / 100) * d["annual_inc"] * 0.3))
                            r = d["int_rate"] / 100 / 12
                            d["installment"] = d.get("installment", round(d["loan_amnt"] * r / (1 - (1 + r) ** (-d["term"]))))
                            rows_for_model.append(d)

                        pds = predict_batch(model, rows_for_model, feature_names)

                    # Add results
                    results_df = batch_df.copy()
                    results_df["predicted_pd"] = pds
                    results_df["credit_score"] = [pd_to_score(p) for p in pds]
                    results_df["risk_level"] = [risk_label(p) for p in pds]
                    results_df["expected_loss"] = pds * LGD * results_df["loan_amnt"]

                    # Summary
                    total_exposure = results_df["loan_amnt"].sum()
                    total_el = results_df["expected_loss"].sum()
                    high_risk_count = (pds > 0.30).sum()
                    avg_pd = pds.mean()

                    st.write("")
                    st.html(f"""
<div class="metric-grid" style="grid-template-columns: 1fr 1fr;">
    <div class="metric-card big"><div class="metric-label">Total Loans Processed</div>
        <div class="metric-value big">{len(results_df):,}</div></div>
    <div class="metric-card big"><div class="metric-label">Average PD</div>
        <div class="metric-value big">{avg_pd*100:.2f}%</div></div>
</div>
<div class="metric-grid" style="grid-template-columns: 1fr 1fr; margin-top: 10px;">
    <div class="metric-card"><div class="metric-label">Total Expected Loss</div>
        <div class="metric-value" style="color: {COLOR_DANGER};">{fmt_money(total_el)}</div></div>
    <div class="metric-card"><div class="metric-label">High Risk Loans</div>
        <div class="metric-value">{high_risk_count} ({high_risk_count/len(results_df)*100:.0f}%)</div></div>
</div>
""")

                    st.write("")
                    st.markdown("**PD Distribution**")
                    st.plotly_chart(make_batch_pd_distribution(pds),
                                    use_container_width=True, config={"displayModeBar": False})

                    # Results table
                    st.markdown("**Results Preview**")
                    st.dataframe(
                        results_df[["loan_amnt", "int_rate", "grade", "fico_range_low",
                                    "predicted_pd", "credit_score", "risk_level", "expected_loss"]].head(20),
                        use_container_width=True,
                    )

                    # Download
                    result_csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Full Results CSV",
                        data=result_csv,
                        file_name=f"credittrace_batch_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        key="dl_batch_results",
                        use_container_width=True,
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.html("""
<div class="empty-state">
    <div class="empty-state-icon">📁</div>
    <div class="empty-state-text">
        Download the template CSV, fill it with your borrower data,
        then upload it here to get batch predictions.
    </div>
</div>
""")


# ============================================================
# TAB 5: MODEL INSIGHTS
# ============================================================
with tab_insights:
    st.html('<div class="section-title">Model Insights</div>')
    st.html('<div class="section-sub">Understand model behavior, threshold trade-offs, and cost analysis.</div>')

    # Generate a synthetic validation set for ROC / cost analysis
    @st.cache_data(show_spinner=False)
    def generate_validation_set(_model, feature_names_tuple: tuple, n: int = 500) -> tuple:
        """Generate a synthetic validation set covering a wide range of profiles."""
        feature_names = list(feature_names_tuple)
        rng = np.random.default_rng(42)
        rows = []
        for _ in range(n):
            grade = rng.choice(list("ABCDEFG"), p=[0.18, 0.28, 0.26, 0.15, 0.08, 0.04, 0.01])
            grade_idx = "ABCDEFG".index(grade)
            fico = int(rng.normal(700 - grade_idx * 15, 30))
            fico = max(610, min(845, fico))
            rate = float(np.clip(rng.normal(7 + grade_idx * 3.5, 2), 5, 30))
            dti = float(np.clip(rng.normal(15 + grade_idx * 3, 5), 0, 40))
            loan_amnt = int(rng.choice([5000, 10000, 15000, 20000, 25000, 30000]))
            annual_inc = int(np.clip(rng.normal(70000 - grade_idx * 5000, 20000), 15000, 300000))
            revol_util = float(np.clip(rng.normal(30 + grade_idx * 10, 15), 0, 120))
            term = int(rng.choice([36, 60]))
            d = {
                "loan_amnt": loan_amnt, "int_rate": rate, "annual_inc": annual_inc,
                "dti": dti, "fico_range_low": fico, "fico_range_high": fico + 4,
                "revol_util": revol_util, "emp_length": int(rng.integers(0, 11)),
                "term": term, "grade": grade, "sub_grade": f"{grade}3",
                "home_ownership": str(rng.choice(["RENT", "MORTGAGE", "OWN"])),
                "purpose": str(rng.choice(["debt_consolidation", "credit_card", "home_improvement"])),
                "delinq_2yrs": 0, "inq_last_6mths": 0, "mths_since_last_delinq": 60,
                "open_acc": 10, "pub_rec": 0, "total_acc": 20,
                "application_type": "Individual",
                "mort_acc": 2 if rng.random() > 0.5 else 0,
                "pub_rec_bankruptcies": 0, "credit_history_months": 120,
                "verification_status": "Verified",
            }
            d["revol_bal"] = round((revol_util / 100) * annual_inc * 0.3)
            r = rate / 100 / 12
            d["installment"] = round(loan_amnt * r / (1 - (1 + r) ** (-term)))
            rows.append(d)
        pds = predict_batch(_model, rows, feature_names)
        # Simulate ground truth using PD as a Bernoulli probability (calibrated)
        y_true = (rng.random(n) < pds).astype(int)
        return pds, y_true

    with st.spinner("Generating validation set..."):
        val_pds, val_y = generate_validation_set(model, tuple(feature_names), n=800)

    from sklearn.metrics import roc_curve, roc_auc_score

    fpr, tpr, thresholds = roc_curve(val_y, val_pds)
    auc_val = roc_auc_score(val_y, val_pds)

    st.markdown("**ROC Curve**")
    st.caption(f"Computed on {len(val_pds)} synthetic validation samples (calibrated Bernoulli).")
    col_roc, col_metrics = st.columns([1.4, 1])
    with col_roc:
        st.plotly_chart(make_roc_curve(fpr, tpr, auc_val),
                        use_container_width=True, config={"displayModeBar": False})
    with col_metrics:
        st.html(f"""
<div class="metric-card big" style="margin-top: 1rem;">
    <div class="metric-label">Validation AUC</div>
    <div class="metric-value big" style="color: {COLOR_SUCCESS};">{auc_val:.4f}</div>
    <div class="metric-sub">On synthetic data</div>
</div>
<div class="metric-card big" style="margin-top: 10px;">
    <div class="metric-label">Production AUC</div>
    <div class="metric-value big" style="color: {COLOR_ACCENT};">{AUC}</div>
    <div class="metric-sub">From cross-validation</div>
</div>
""")

    # Threshold simulator
    st.write("")
    st.markdown("**Threshold & Cost Analysis**")
    st.caption("Adjust the cost of errors to find the optimal decision threshold.")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        cost_fp = st.slider("Cost of False Positive (rejecting good borrower) $", 100, 10000, 500, 100, key="cost_fp")
    with col_c2:
        cost_fn = st.slider("Cost of False Negative (accepting bad borrower) $", 500, 50000, 5000, 500, key="cost_fn")

    # Compute confusion matrix for each threshold
    thresh_grid = np.linspace(0.01, 0.99, 99)
    total_costs = []
    for t in thresh_grid:
        y_pred = (val_pds >= t).astype(int)
        tp = int(((y_pred == 1) & (val_y == 1)).sum())
        fp = int(((y_pred == 1) & (val_y == 0)).sum())
        fn = int(((y_pred == 0) & (val_y == 1)).sum())
        tn = int(((y_pred == 0) & (val_y == 0)).sum())
        total_costs.append(fp * cost_fp + fn * cost_fn)
    total_costs = np.array(total_costs)
    optimal_idx = int(np.argmin(total_costs))
    optimal_t = float(thresh_grid[optimal_idx])

    st.write("")
    st.plotly_chart(make_cost_curve(thresh_grid, total_costs, optimal_t),
                    use_container_width=True, config={"displayModeBar": False})

    # Confusion matrix at optimal threshold
    y_pred_opt = (val_pds >= optimal_t).astype(int)
    tp = int(((y_pred_opt == 1) & (val_y == 1)).sum())
    fp = int(((y_pred_opt == 1) & (val_y == 0)).sum())
    fn = int(((y_pred_opt == 0) & (val_y == 1)).sum())
    tn = int(((y_pred_opt == 0) & (val_y == 0)).sum())

    st.write("")
    col_cm, col_stats = st.columns([1.2, 1])
    with col_cm:
        st.markdown(f"**Confusion Matrix at threshold = {optimal_t:.2f}**")
        st.plotly_chart(make_confusion_heatmap(tp, fp, fn, tn),
                        use_container_width=True, config={"displayModeBar": False})
    with col_stats:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        st.html(f"""
<div class="metric-card big" style="margin-top: 1rem;">
    <div class="metric-label">Optimal Threshold</div>
    <div class="metric-value big" style="color: {COLOR_SUCCESS};">{optimal_t:.2f}</div>
</div>
<div class="metric-grid" style="margin-top: 10px;">
    <div class="metric-card"><div class="metric-label">Precision</div>
        <div class="metric-value">{precision*100:.1f}%</div></div>
    <div class="metric-card"><div class="metric-label">Recall</div>
        <div class="metric-value">{recall*100:.1f}%</div></div>
    <div class="metric-card"><div class="metric-label">Accuracy</div>
        <div class="metric-value">{accuracy*100:.1f}%</div></div>
    <div class="metric-card"><div class="metric-label">Total Cost</div>
        <div class="metric-value" style="color: {COLOR_DANGER};">{fmt_money(total_costs[optimal_idx])}</div></div>
</div>
""")

    # Model info
    st.write("")
    st.markdown("**Model Card**")
    st.html(f"""
<div class="input-summary">
    <div class="input-row"><span class="input-label">Algorithm</span><span class="input-value">LightGBM v5</span></div>
    <div class="input-row"><span class="input-label">AUC (CV)</span><span class="input-value">{AUC}</span></div>
    <div class="input-row"><span class="input-label">KS Statistic</span><span class="input-value">{KS}</span></div>
    <div class="input-row"><span class="input-label">Gini</span><span class="input-value">0.4573</span></div>
    <div class="input-row"><span class="input-label">Features</span><span class="input-value">{len(feature_names)}</span></div>
    <div class="input-row"><span class="input-label">Trees</span><span class="input-value">{model.num_trees():,}</span></div>
    <div class="input-row"><span class="input-label">Training Rows</span><span class="input-value">1,078,479</span></div>
    <div class="input-row"><span class="input-label">Default Rate</span><span class="input-value">19.98%</span></div>
    <div class="input-row"><span class="input-label">Calibration</span><span class="input-value" style="color: {COLOR_SUCCESS};">✓ Perfect (19.98% = 19.98%)</span></div>
    <div class="input-row"><span class="input-label">Leakage</span><span class="input-value" style="color: {COLOR_SUCCESS};">✓ Removed</span></div>
    <div class="input-row"><span class="input-label">Fairness</span><span class="input-value" style="color: {COLOR_SUCCESS};">✓ addr_state removed</span></div>
</div>
""")


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.html(
    '<p style="text-align: center; color: #64748b; font-size: 0.78rem; line-height: 1.8;">'
    '<strong style="color: #e8eef5;">CreditTrace</strong> — Credit Risk, Engineered.<br>'
    "The complete ML audit trail for credit risk.<br>"
    '<a href="https://github.com/erfan2mohammadi22/CreditTrace" target="_blank" '
    'style="color: #00d4ff; text-decoration: none; margin: 0 8px;">GitHub</a> · '
    '<a href="https://erfan2mohammadi22.github.io/CreditTrace/" target="_blank" '
    'style="color: #00d4ff; text-decoration: none; margin: 0 8px;">WASM Version</a>'
    "</p>"
)