# The Process

> Six model versions. One critical discovery. A deliberate 1% AUC sacrifice.

This document tells the complete story of how CreditTrace went from a baseline model to a production-ready system, including the mistakes, the discoveries, and the trade-offs along the way.

The story matters because **the journey is more valuable than the destination**.

---

## Table of Contents

1. [The Baseline](#1-the-baseline)
2. [Optuna Tuning](#2-optuna-tuning)
3. [The Discovery](#3-the-discovery)
4. [The Decision](#4-the-decision)
5. [The Ablation Study](#5-the-ablation-study)
6. [The Golden Finding](#6-the-golden-finding)
7. [Verification](#7-verification)
8. [The Financial Layer](#8-the-financial-layer)
9. [Deployment](#9-deployment)
10. [Version History](#10-version-history)
11. [Lessons Learned](#11-lessons-learned)

---

## 1. The Baseline

Every project starts somewhere. Ours started with three models and a simple question: what does the data say?

We trained three models on the full dataset, all 47 engineered features:

| Model | AUC | Notes |
|---|---|---|
| Logistic Regression | 0.7146 | Baseline |
| XGBoost | 0.7295 | Depth-wise tree growth |
| **LightGBM** | **0.7349** | Leaf-wise, native categorical support |

LightGBM won clearly. Its native categorical support and histogram-based splits worked well with our mixed data.

But 0.7349 was not enough. We wanted to know what tuning could do.

---

## 2. Optuna Tuning

We ran 30 trials with Optuna, using a 25% subsample and 3-fold cross-validation for speed. Median pruning stopped weak trials early.

**Best trial: #17**

```python
{
    "learning_rate": 0.01106,
    "num_leaves": 66,
    "max_depth": 12,
    "min_child_samples": 177,
    "feature_fraction": 0.6006,
    "bagging_fraction": 0.7481,
    "bagging_freq": 1,
    "lambda_l1": 0.6664,
    "lambda_l2": 0.00118,
    "min_gain_to_split": 0.1627,
}
```

**v2 AUC = 0.7357** (hit the 2000-round ceiling)

The improvement was modest — about +0.0008 AUC — but every basis point counts in credit risk.

---

## 3. The Discovery

We re-trained on the full dataset with `max_rounds=4000` and early stopping. This gave us **v3 with AUC = 0.7363**.

Something was odd though. All 5 folds hit the 4000-round ceiling; early stopping never triggered.

But the real discovery came from **SHAP**.

We ran SHAP on v3 to understand what the model had learned. The result was striking:

| Rank | Feature | SHAP Importance |
|---|---|---|
| 1 | `sub_grade` | 0.27 |
| 2 | `term` | 0.18 |
| **3** | **`issue_year`** | **0.17** |
| 4 | `grade` | 0.13 |
| 5 | `dti` | 0.11 |
| 6 | `home_ownership` | 0.11 |
| **7** | **`addr_state`** | **0.10** |
| ... | ... | ... |
| 14 | `int_rate` | 0.05 |

`issue_year` was the **third most important feature**. But it has nothing to do with the borrower.

The SHAP dependence plot made it worse:

```
issue_year:
    2007–2014: around zero
    2015–2017: positive (higher default)
    2018:      sharp negative drop
```

This is a **macro-economic pattern**, not a borrower signal. The model learned that loans issued in 2015–2017 defaulted more often — but that's a statement about the economy, not about the person.

If this model were deployed in 2019, it would fail on 2020's COVID-driven defaults.

`addr_state` was also problematic. Its SHAP variance across 50 states was extremely high (noisy), and using geographic location in credit decisions has well-documented disparate-impact implications.

**This was temporal leakage and fairness risk.** Both had to go.

---

## 4. The Decision

Removing `issue_year`, `issue_month`, `issue_quarter`, and `addr_state` meant:

- From 47 features down to 43 (then 41 after removing `target` and `loan_status` from the model input)
- From **AUC = 0.7363** to **AUC = 0.7286**
- A cost of roughly **1% AUC**

That is millions of dollars at scale. But three reasons made the trade-off worthwhile.

### 1. Production-readiness

A model dependent on calendar year cannot generalize to future data. It would fail silently, with no warning, on the first year that behaves differently from the training set.

### 2. Fairness

A model dependent on geography will treat borrowers differently based on where they live. That is not credit risk modeling; that is proxy discrimination.

### 3. Scientific integrity

Keeping the leakage would mean reporting a number we knew was inflated. That is not engineering; that is self-deception.

We removed the features. We re-trained. **AUC = 0.7286.**

---

## 5. The Ablation Study

Something was still wrong.

The v4 model (with 4000 rounds) hit the ceiling in **every fold**. Early stopping never triggered. Either the learning rate was too high, or we needed more rounds.

We designed a controlled experiment to isolate the bottleneck.

| Version | `learning_rate` | `max_rounds` | AUC | `avg_best_iter` |
|---|---|---|---|---|
| v4 | 0.011 | 4,000 | 0.7286 | 3,989 (ceiling) |
| **v5** | **0.011** | **8,000** | **0.7287** | **4,034** |
| v6 | 0.025 | 8,000 | 0.7283 | 1,855 |

**The finding:**

With a higher ceiling, v5 finally triggered early stopping at 4,034 rounds. The bottleneck was the total training budget, not the learning rate.

More importantly: v5 and v6 landed at the same AUC, despite v6 using **50% fewer trees**. The dataset had a ceiling. We had reached it.

We selected v5 as the final model: the best AUC, with meaningful early stopping behavior.

---

## 6. The Golden Finding

After removing leakage features, we re-ran SHAP on v5. Something remarkable had happened.

| Feature | v3 Rank | v5 Rank | Change |
|---|---|---|---|
| `sub_grade` | 1 | 1 | — |
| `term` | 2 | 2 | — |
| `issue_year` | **3** | **removed** | — |
| `grade` | 4 | 3 | +1 |
| `dti` | 5 | 4 | +1 |
| `home_ownership` | 6 | 5 | +1 |
| `addr_state` | **7** | **removed** | — |
| `inq_last_6mths` | 8 | 9 | -1 |
| `fico_range_low` | 9 | 7 | +2 |
| `mort_acc` | 10 | 8 | +2 |
| **`int_rate`** | **14** | **6** | **+8** |

**`int_rate` jumped from rank 14 to rank 6.**

Its importance nearly doubled: from 0.05 to 0.098.

### Why this matters

In v3, the model relied on `issue_year` and `addr_state` — proxy signals that happened to correlate with outcomes in the training data.

In v5, freed from those crutches, the model leaned on `int_rate` — the interest rate itself. And the interest rate **is** a risk signal. Banks set it based on their own assessment of default probability.

The model had stopped cheating and started learning.

---

## 7. Verification

Before deploying, we needed to check that the model was telling the truth.

AUC alone is not enough. For a financial model, **calibration** matters even more.

### The calibration check

```
Mean predicted PD:     19.98%
Actual default rate:   19.98%
```

**Perfectly calibrated.**

If the model says 20%, then 20% of those borrowers actually default. This is critical for any financial application.

A model with high AUC but poor calibration would give the right ranking but the wrong dollar amounts. CreditTrace has both.

---

## 8. The Financial Layer

AUC does not answer the question a bank actually asks: *"How much money will we lose?"*

We computed **Expected Loss** using the industry-standard formula:

```
EL = PD × LGD × EAD
```

Where:
- **PD** — Probability of Default (from the model)
- **LGD** — Loss Given Default (45%, industry standard)
- **EAD** — Exposure at Default (loan amount)

### Results by grade

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

The suggested rate is derived from the model's own predictions, not from a separate pricing model. It is the model's output translated into the language a bank actually uses: basis points of interest rate.

---

## 9. Deployment

We deployed the model in two fundamentally different architectures.

### Client-side (WASM)

The trained LightGBM model was exported to ONNX and runs entirely in the browser.

Challenges we solved:
- **Multi-threading in the browser** — required `SharedArrayBuffer`, which requires COOP/COEP headers, which we injected via a Service Worker (`coi-serviceworker.js`)
- **Off-main-thread inference** — required a Web Worker for the ONNX session, so the UI never froze
- **Lazy initialization** — the 30 MB model downloads only when the user actually wants custom analysis
- **Pre-computed presets** — 5 common profiles are pre-computed and shipped as a 6 KB JSON, so casual users get instant results

### Server-side (Streamlit)

The same model runs on Streamlit Cloud as a full analysis tool with:
- Real SHAP (not approximation)
- Counterfactual analysis
- Portfolio simulation
- Batch prediction
- Cost-sensitive threshold optimization

Both deployments share the same v5 model and the same AUC. The choice between them is a choice between privacy and features.

---

## 10. Version History

| Version | Features | AUC | Notes |
|---|---|---|---|
| v1 | 47 | 0.7349 | LightGBM baseline, no tuning |
| v2 | 47 | 0.7357 | Optuna Trial #17, hit 2000-round ceiling |
| v3 | 47 | 0.7363 | **With leakage**, hit 4000-round ceiling |
| v4 | 43 | 0.7286 | Leakage removed, hit 4000-round ceiling |
| **v5** | **41** | **0.7287** | **Final model**, early stopping at 4,034 |
| v6 | 41 | 0.7283 | 50% fewer trees, same AUC |

The chosen model is **v5**.

---

## 11. Lessons Learned

### 1. The best model is not the one with the highest AUC

A model that is production-ready, fair, and calibrated is worth more than one with 1% higher AUC and hidden leakage.

### 2. SHAP is not optional

AUC alone would never have revealed the leakage. Interpretation is what separates engineering from guessing.

### 3. When a model hits a ceiling, check the data first

The Ablation Study proved the bottleneck was not hyperparameters, it was the signal in the data itself.

### 4. Calibration is what makes a model useful

AUC = 0.73 with perfect calibration is worth more to a bank than AUC = 0.80 with a miscalibrated score.

### 5. Deployment is half the work

The other half is documentation, interpretation, and honest reporting. A model that cannot be explained is a model that cannot be trusted.

---

## See Also

- [README.md](README.md) — Project overview and results
- [CONCEPTS.md](CONCEPTS.md) — Explanations of every technique used

---

**Credit Risk, Engineered.**