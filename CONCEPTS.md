# Concepts

> Thirteen core ML concepts used in CreditTrace, explained in plain language.

Whether you are learning ML, preparing for interviews, or just curious, this document explains every technique used in the project.

---

## Table of Contents

1. [Credit Risk — The Foundation](#1-credit-risk--the-foundation)
2. [Parquet vs CSV — Storing Data Efficiently](#2-parquet-vs-csv--storing-data-efficiently)
3. [Cross-Validation — Trusting Your Score](#3-cross-validation--trusting-your-score)
4. [Early Stopping — Knowing When to Stop](#4-early-stopping--knowing-when-to-stop)
5. [SHAP — Opening the Black Box](#5-shap--opening-the-black-box)
6. [Calibration — When Probabilities Mean Something](#6-calibration--when-probabilities-mean-something)
7. [Expected Loss — From Probability to Money](#7-expected-loss--from-probability-to-money)
8. [Risk-Based Pricing — Charging for Risk](#8-risk-based-pricing--charging-for-risk)
9. [Data Leakage — The Silent Killer](#9-data-leakage--the-silent-killer)
10. [Ablation Study — Isolating What Matters](#10-ablation-study--isolating-what-matters)
11. [Fairness in ML — Who Gets Hurt?](#11-fairness-in-ml--who-gets-hurt)
12. [LightGBM vs XGBoost vs CatBoost](#12-lightgbm-vs-xgboost-vs-catboost)
13. [ONNX — Running Models Anywhere](#13-onnx--running-models-anywhere)

---

## 1. Credit Risk — The Foundation

**Credit risk** is the possibility that a borrower fails to repay a loan. For a lender, every approved loan is a bet: will this person pay us back, with interest?

Traditional credit scoring (like FICO) uses a fixed formula. Modern approaches use machine learning to combine dozens or hundreds of signals — income, credit history, debt-to-income ratio, loan purpose — into a single probability of default.

The output of a credit risk model is a **Probability of Default (PD)**: a number between 0 and 1 representing how likely a borrower is to default.

In CreditTrace, PD is the model's output, and everything else (Expected Loss, risk-based pricing, decision thresholds) is built on top of it.

---

## 2. Parquet vs CSV — Storing Data Efficiently

**CSV** is text: every value is a string, every row is a line. Simple, universal, but slow and fat. Our Lending Club CSV was **392 MB** compressed.

**Parquet** is binary and columnar. Instead of storing rows, it stores columns together. When you only need three columns out of 40, Parquet reads only those three. It also preserves data types and compresses naturally.

| | CSV | Parquet |
|---|---|---|
| Format | Text | Binary |
| Layout | Row-based | Columnar |
| Typed | No | Yes |
| Compression | External | Built-in |
| Read speed | Slow | Fast |
| File size (our data) | 392 MB (gzipped) | 67 MB |

Our data shrank from 392 MB (CSV.gz) to **67 MB (Parquet)**, and loads in seconds instead of minutes.

---

## 3. Cross-Validation — Trusting Your Score

If you train a model on all your data and test on the same data, it will look perfect, but it has memorized the answers. A single train/test split helps, but the score depends on which rows happened to land in the test set.

**K-Fold Cross-Validation** solves this. Split the data into K parts. Train K times, each time holding out a different part for validation. Average the K scores.

```
Final score = mean(AUC₁, AUC₂, ..., AUC_K)
```

**Stratified** K-Fold goes further: it preserves the class balance in each fold, which matters when the positive class is rare.

CreditTrace uses **Stratified 5-Fold CV** on all model evaluations. Every AUC number you see is an average across five independent validation sets.

---

## 4. Early Stopping — Knowing When to Stop

Gradient boosting builds trees one at a time. Each tree tries to correct the mistakes of the previous ones. In theory, more trees equals better fit. In practice, after a certain point the model starts memorizing noise.

**Early stopping** watches the validation score after every tree. If the score has not improved in *N* consecutive trees, training stops.

```
Stop when validation AUC doesn't improve for N rounds
```

In CreditTrace, we used `early_stopping_rounds = 100`. If the model does not improve for 100 trees, it stops.

In v4, early stopping never triggered — the model hit the 4000-round ceiling. In v5, with an 8000-round ceiling, it stopped naturally at 4,034 rounds. That discovery was the whole point of the Ablation Study.

---

## 5. SHAP — Opening the Black Box

A gradient boosted model with 4,000 trees is not something you can read by staring at it. **SHAP** (SHapley Additive exPlanations) gives each feature a numeric contribution to each prediction.

The idea comes from game theory: treat the model output as a "payout" and each feature as a "player." How much did each player contribute to the final outcome?

- **Positive SHAP** — the feature pushed the prediction toward default
- **Negative SHAP** — the feature pushed the prediction away from default
- **Absolute SHAP** — how much influence the feature had

SHAP provides both global importance (which features matter most overall) and local explanations (why this specific borrower got this specific score).

**In CreditTrace**, SHAP is how we discovered the temporal leakage. Without it, `issue_year` would still be in the model, silently inflating the AUC.

---

## 6. Calibration — When Probabilities Mean Something

A model can rank borrowers correctly (high AUC) and still be **miscalibrated**: it might say "30% default probability" when the true rate is 15%. That makes it useless for any financial calculation.

**Calibration** means: when the model says X%, then X% of those borrowers actually default.

```
mean(predicted PD) ≈ actual default rate
```

In CreditTrace: the mean predicted PD is **19.98%**, and the actual default rate in the data is **19.98%**. Perfectly calibrated.

**Why this matters:** Calibration is what makes Expected Loss work. If PD is wrong on average, every dollar figure built on it is wrong too.

---

## 7. Expected Loss — From Probability to Money

A bank does not just want to know the probability of default. It wants to know **how much money it will lose**. That is Expected Loss.

```
EL = PD × LGD × EAD
```

- **PD** — Probability of Default (from the model)
- **LGD** — Loss Given Default: the fraction of exposure lost if default happens
- **EAD** — Exposure at Default: the outstanding balance at the moment of default

We use the industry-standard LGD of **45%** and treat EAD as the loan amount (a simplification).

In CreditTrace, Expected Loss ranges from **$374 for grade A** to **$4,576 for grade G** — a 12× difference on the same $10,000 loan.

---

## 8. Risk-Based Pricing — Charging for Risk

If a high-risk borrower is charged the same rate as a low-risk one, the low-risk borrower subsidizes the high-risk one. That is not sustainable.

**Risk-based pricing** means setting the interest rate to cover the expected loss, plus operating costs and profit margin.

```
min_rate = cost_of_funds + EL_rate + operating_cost + profit_margin
```

With `COF = 2%`, `OC = 1%`, `PM = 3%`:

| Grade | Suggested Rate |
|---|---|
| A | 8.7% |
| B | 12.0% |
| C | 16.1% |
| D | 19.7% |
| E | 23.3% |
| F | 26.3% |
| G | 28.4% |

This is the model's output translated into the language a bank actually uses: basis points of interest rate.

---

## 9. Data Leakage — The Silent Killer

**Leakage** happens when a feature contains information that would not be available at prediction time, but is present in the training data.

Common types:

- **Temporal leakage** — features derived from the future (issue date, year)
- **Target leakage** — features computed using the target (recovery amount)
- **Group leakage** — train and test rows share a group (same person)

Leakage inflates your validation score and produces a model that fails in production. It is often invisible to AUC — you need SHAP and thought to catch it.

**In CreditTrace**, our `issue_year` feature was the third most important in v3. It "helped" the model learn that 2015–2017 had higher default rates. That is a macro-economic cycle, not a borrower signal. Removing it cost 1% AUC but produced a model that could survive contact with the future.

See [PROCESS.md](PROCESS.md) for the full story.

---

## 10. Ablation Study — Isolating What Matters

An **ablation study** is a controlled experiment: change one thing at a time, keep everything else fixed, measure the effect.

The name comes from neuroscience: removing (ablating) parts of a brain to see what each part does.

In CreditTrace, v4 hit the 4000-round ceiling in every fold. Two hypotheses:

1. The learning rate was too high → try a lower one
2. The round budget was too small → try more rounds

We designed v5 (lower LR, more rounds) and v6 (higher LR, more rounds) to isolate the effect. Result: the bottleneck was the round budget, not the learning rate.

After 8000 rounds, v5 stopped naturally at 4,034. The model had found its optimum. Additional rounds would have overfit.

---

## 11. Fairness in ML — Who Gets Hurt?

Machine learning models can perpetuate or amplify discrimination. In credit risk, using features like geography, race, or gender proxies can produce **disparate impact**: treating some groups systematically worse.

Common approaches:

- **Pre-processing** — remove or transform sensitive features
- **In-processing** — add fairness constraints to training
- **Post-processing** — adjust thresholds per group

In CreditTrace, we took the simplest honest path: remove `addr_state` because its SHAP variance was extremely high and geography has well-documented fairness implications in lending.

A full fairness audit (disparate impact, equal opportunity across protected groups) requires sensitive attributes we do not have. We document this as a limitation, not a solved problem.

---

## 12. LightGBM vs XGBoost vs CatBoost

All three are gradient boosting libraries. They share the same core idea — sequentially build trees that correct each other's errors — but differ in implementation details.

| | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Depth-wise | Leaf-wise | Symmetric |
| Speed on large data | Moderate | Fast | Fast |
| Memory usage | High | Low | Moderate |
| Categorical handling | Encoding needed | Native | Native |
| Typical use case | General purpose | Large datasets | Heavy categorical |

**XGBoost** is depth-wise: it grows all branches of a tree to the same depth. Very well-documented, widely used, historically the standard.

**LightGBM** is leaf-wise: it grows the leaf with the highest loss reduction, regardless of depth. This makes it faster on large data and lower-memory.

**CatBoost** uses ordered boosting and special handling of categoricals, often best when categorical features dominate.

In our tests, LightGBM consistently outperformed XGBoost on this dataset (0.7349 vs. 0.7295 at baseline), primarily due to native categorical handling and the leaf-wise growth strategy.

**Rule of thumb:** LightGBM for speed and large data, XGBoost for stability and ecosystem, CatBoost for heavy categorical use cases.

---

## 13. ONNX — Running Models Anywhere

**ONNX** (Open Neural Network Exchange) is an open format for representing machine learning models. It is designed so a model trained in one framework can run in another.

Why it matters:

- A LightGBM model can be exported to ONNX
- The same ONNX file runs in Python, C++, JavaScript, mobile, or the browser
- No need to reimplement the algorithm for each platform

**In CreditTrace**, the trained LightGBM model was converted to ONNX and loaded in the browser via WebAssembly. The same file that predicts in Python predicts in your browser, verified numerically (max difference: 5.47 × 10⁻⁷).

**Challenges we solved:**

- Multi-threading in the browser requires `SharedArrayBuffer`, which requires COOP/COEP headers, which we injected via a Service Worker
- Off-main-thread inference requires a Web Worker, so the UI never froze during model initialization
- The 30 MB model downloads lazily, only when the user actually wants custom analysis

This is why the documentation site's demo works with no server. The model lives in your browser tab. No data is sent anywhere. It runs in about 1.2 milliseconds per prediction.

---

## See Also

- [README.md](README.md) — Project overview and results
- [PROCESS.md](PROCESS.md) — The journey from v1 to v6

---

**Credit Risk, Engineered.**