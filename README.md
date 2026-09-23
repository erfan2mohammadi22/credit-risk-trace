# CreditTrace

> **Credit Risk, Engineered.**
> The complete ML audit trail for credit risk.

An end-to-end machine learning pipeline for credit default prediction. Trained on **1.3 million real loans** from Lending Club, rigorously validated with 5-fold cross-validation, and deployed in two fundamentally different architectures: a client-side WASM app and a server-side Streamlit tool.

[![AUC](https://img.shields.io/badge/AUC-0.7287-00d4ff?style=flat-square)](https://github.com/erfan2mohammadi22/credit-risk-trace)
[![KS](https://img.shields.io/badge/KS-0.3322-8b5cf6?style=flat-square)](https://github.com/erfan2mohammadi22/credit-risk-trace)
[![Gini](https://img.shields.io/badge/Gini-0.4573-00e676?style=flat-square)](https://github.com/erfan2mohammadi22/credit-risk-trace)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## Two Live Demos

CreditTrace is deployed in two fundamentally different architectures. Same model, same AUC, different trade-offs.

| | Documentation Site | Analysis Tool |
|---|---|---|
| **URL** | [erfan2mohammadi22.github.io/credit-risk-trace](https://erfan2mohammadi22.github.io/credit-risk-trace/) | [credit-risk-trace.streamlit.app](https://credit-risk-trace.streamlit.app/) |
| **Architecture** | Client-side (WASM) | Server-side (Streamlit Cloud) |
| **Focus** | Education · Model Card · Journey | Analysis · Portfolio · Batch |
| **Privacy** | 100% — no data leaves your device | Data sent to server |
| **First Load** | ~20 seconds (downloads the model) | Instant |
| **Speed** | Sub-second after warmup | Depends on server |
| **Cost** | Free forever | Free tier (1 concurrent user) |
| **Offline** | Works after first load | Requires internet |
| **Best for** | Understanding the model | Using the model |

### Documentation Site Features

- **Home** — Project overview, key discoveries, SHAP preview
- **Model** — Full model card, version history, calibration check
- **Data** — Data story, EDA charts, feature engineering
- **Process** — The journey from v1 to v6, leakage discovery, ablation study
- **Concepts** — 13 core ML concepts explained in plain language
- **Demo** — Interactive prediction in your browser (client-side)

### Analysis Tool Features

- **Custom Analysis** — Real-time prediction with real SHAP and counterfactual analysis
- **Portfolio Simulator** — Adjust grade distribution and see total expected loss
- **Compare Borrowers** — Side-by-side A/B analysis with visual diffs
- **Batch Prediction** — Upload CSV and get predictions for all rows
- **Model Insights** — ROC curve, cost matrix, confusion matrix, threshold optimizer

---

## Model Performance

| Metric | Value |
|---|---|
| **Algorithm** | LightGBM v5 (GBDT) |
| **AUC** | **0.7287** |
| **KS** | **0.3322** |
| **Gini** | **0.4573** |
| **Features** | 41 |
| **Trees** | 4,235 |
| **Training rows** | 1,078,479 |
| **Test rows** | 269,620 |
| **Default rate** | 19.98% |
| **Calibration** | Perfect (PD = 19.98% = actual) |
| **Leakage** | None |
| **Fairness** | `addr_state` removed |

Cross-validated with **Stratified 5-Fold CV** and early stopping at 4,034 rounds.

---

## The Story: Why AUC = 0.7287 (and not 0.7363)

The journey from a naive model to a production-ready one involved one critical discovery and one uncomfortable decision.

### The discovery

The initial model (v3) achieved **AUC = 0.7363** with 47 features. But SHAP analysis revealed something suspicious:

| Rank | Feature | SHAP Importance |
|---|---|---|
| 1 | `sub_grade` | 0.27 |
| 2 | `term` | 0.18 |
| **3** | **`issue_year`** | **0.17** (leakage) |
| 4 | `grade` | 0.13 |
| 5 | `dti` | 0.11 |
| **6** | **`addr_state`** | **0.10** (fairness) |

`issue_year` had become the third most important feature, but it has **nothing to do with the borrower**. The model was learning macro-economic conditions (2015–2017 had higher default rates), which is **temporal leakage**.

`addr_state` was also problematic: high SHAP variance across 50 states and clear fairness concerns.

### The decision

Removing these features cost roughly **1% of AUC** (0.7363 to 0.7286). But it was the right call for three reasons:

1. **Production-readiness** — a model dependent on calendar year cannot generalize
2. **Fairness** — geography should not drive credit decisions
3. **Scientific integrity** — keeping leakage would mean lying to ourselves

### The golden finding

After removal, we re-ran SHAP. The result was striking:

| Feature | v3 Rank | v5 Rank |
|---|---|---|
| `sub_grade` | 1 | 1 |
| `term` | 2 | 2 |
| `issue_year` | **3** | removed |
| `grade` | 4 | 3 |
| `dti` | 5 | 4 |
| `home_ownership` | 6 | 5 |
| `addr_state` | **7** | removed |
| **`int_rate`** | **14** | **6** |

`int_rate` jumped from rank 14 to rank 6, doubling its importance (0.05 to 0.098). Freed from leakage, the model learned to rely on the interest rate, which is itself a reflection of credit risk. **This is exactly what a credit model should do.**

See [PROCESS.md](PROCESS.md) for the complete journey.

---

## Beyond ML: Expected Loss

AUC alone does not answer the question a bank actually asks: *"How much money will we lose?"*

CreditTrace computes **Expected Loss** using the industry-standard formula:

```
EL = PD × LGD × EAD
```

| Grade | PD | Expected Loss | Suggested Rate |
|---|---|---|---|
| A | 6.0% | $374 | 8.7% |
| B | 13.4% | $817 | 12.0% |
| C | 22.4% | $1,488 | 16.1% |
| D | 30.4% | $2,186 | 19.7% |
| E | 38.4% | $3,129 | 23.3% |
| F | 45.1% | $3,965 | 26.3% |
| G | 49.7% | $4,576 | 28.4% |

**Expected loss varies 12× between grade A and grade G.**

---

## Project Structure

```
credit-risk-project/
├── data/                          # Lending Club (gitignored)
│   ├── raw/
│   └── processed/
├── models/                        # Trained .joblib (gitignored)
├── reports/                       # Metrics + figures
│   ├── figures/
│   └── archive_v1..v6/
├── src/                           # Training pipeline (Python)
│   ├── data_loader.py
│   ├── preprocessing.py           # v4 pipeline
│   ├── eda.py
│   ├── train.py                   # Logistic + XGBoost
│   ├── tune.py                    # Optuna
│   ├── retrain.py                 # v4
│   ├── ablation.py                # v5 + v6
│   ├── shap_analysis.py
│   ├── expected_loss.py
│   ├── export_onnx.py
│   └── export_presets.py
├── docs/                          # Documentation site (GitHub Pages)
│   ├── index.html
│   ├── model.html
│   ├── data.html
│   ├── process.html
│   ├── concepts.html
│   ├── demo.html
│   ├── css/
│   ├── js/
│   ├── onnx/                      # model.onnx + metadata + WASM
│   └── figures/
├── streamlit/                     # Analysis tool (Streamlit Cloud)
│   ├── app.py
│   ├── model.joblib
│   ├── metadata.json
│   ├── requirements.txt
│   └── .streamlit/config.toml
├── README.md                      # This file
├── PROCESS.md                     # Journey v1 to v6
├── CONCEPTS.md                    # 13 core ML concepts
├── LICENSE
└── requirements.txt
```

---

## How to Run

### Option 1: Run the Documentation Site Locally

```bash
git clone https://github.com/erfan2mohammadi22/credit-risk-trace.git
cd credit-risk-trace

cd docs
python -m http.server 8000
```

Then open `http://localhost:8000`.

**Note:** The site needs to be served over HTTP (not `file://`) because it uses Service Worker and Web Worker for ONNX inference.

### Option 2: Run the Streamlit Tool Locally

```bash
pip install -r streamlit/requirements.txt

cd streamlit
streamlit run app.py
```

Then open `http://localhost:8501`.

### Option 3: Reproduce the Training Pipeline

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

# Download Lending Club dataset from Kaggle:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club
# Place accepted_2007_to_2018Q4.csv.gz in data/raw/

python src/data_loader.py
python src/preprocessing.py
python src/eda.py
python src/train.py
python src/ablation.py
python src/shap_analysis.py
python src/expected_loss.py
python src/export_onnx.py
python src/export_presets.py
```

Training requires approximately 30 GB RAM and 2 to 4 hours.

---

## Tech Stack

**Data and ML**

- pandas, numpy, pyarrow
- scikit-learn, LightGBM, XGBoost
- Optuna (hyperparameter tuning)

**Explainability**

- SHAP (TreeExplainer) — both in Python and in the browser

**Deployment**

- Client-side: ONNX Runtime Web (WASM) + Web Worker + Service Worker + SharedArrayBuffer
- Server-side: Streamlit + Plotly
- Hosting: GitHub Pages + Streamlit Cloud

**Visualization**

- matplotlib, seaborn (in reports)
- Plotly (in Streamlit)
- Custom SVG and CSS (in the documentation site)

---

## What This Project Demonstrates

If you are evaluating this as a portfolio piece, here is what to look for.

### ML Engineering

- Leakage detection — SHAP-driven discovery of temporal leakage
- Ablation study — isolating the effect of `learning_rate` versus `max_rounds`
- Calibration — perfectly matching predicted probabilities to actual outcomes
- Multi-version tracking — v1 to v6, all documented
- Feature engineering — 17 engineered features, 3 in the top 15 by SHAP

### Software Engineering

- Dual deployment — same model, two architectures
- Web Worker and Service Worker — off-main-thread inference
- Multi-threading in the browser — via SharedArrayBuffer and COOP/COEP
- Mobile-first responsive design
- Progressive model loading — 2-step progress with lazy initialization

### Financial Engineering

- Expected Loss — PD × LGD × EAD
- Risk-based pricing — from probability to interest rate
- Portfolio simulation — grade distribution to total loss
- Cost-sensitive threshold optimization
- Counterfactual analysis — what would improve this profile?

---

## Limitations

- **AUC = 0.7287** is competitive but not exceptional. Lending Club is a difficult, noisy dataset where realistic ceilings sit around 0.73 to 0.75.
- **LGD is assumed at 45%** — an industry standard, not estimated from recovery data.
- **Fairness analysis** is limited to removing `addr_state`. A full disparate-impact analysis is out of scope.
- **Training data ends in 2018.** A production model would need periodic retraining.
- **The Streamlit free tier** sleeps after 7 days of inactivity, causing a ~30 second cold start.

See [CONCEPTS.md](CONCEPTS.md) for explanations of every technique used.

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Contact

**Erfan Mohammadi**

- GitHub: [@erfan2mohammadi22](https://github.com/erfan2mohammadi22)
- LinkedIn: [erfan2mohammadi22](https://linkedin.com/in/erfan2mohammadi22)
- Email: erfan2mohammadi22@gmail.com
- Telegram: [@ERFANmmmig29](https://t.me/ERFANmmmig29)

---

**Built with Python · LightGBM · ONNX · WASM · Streamlit**
