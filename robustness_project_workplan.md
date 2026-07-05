# Total Workplan — Robustness Evaluation Framework for ML Models

**Total duration:** 17 core days + 2 buffer/stretch days = 19 days
**Structure:** 6 phases, each phase ends with a "done when" checkpoint so you know you're not moving forward on a shaky foundation.
**Note:** Phase 6 was rescoped from a Streamlit dashboard (2 days) to a custom
FastAPI + React web app (5 days) — see Phase 6 below for why the extra time is real,
not padding.

---

## Master checklist (tick as you go)

- [ ] Phase 1 — Setup & data (Day 1–2)
- [ ] Phase 2 — Baseline models (Day 3)
- [ ] Phase 3 — Stress-test engine (Day 4–6)
- [ ] Phase 4 — Evaluation, aggregation, robustness index (Day 7–8)
- [ ] Phase 5 — Visualization & insight generator (Day 9–10)
- [ ] Phase 6 — Custom web app: FastAPI + React (Day 11–15)
- [ ] Phase 7 — Testing, docs, polish (Day 16–17)
- [ ] Stretch — extensions + presentation prep (Day 18–19)

---

## Phase 1: Setup & Data (Day 1–2)

**Day 1 — Environment + repo**
- Create the folder structure (below), init git, set up a virtual environment.
- Install: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `streamlit`.
- Download the **Adult Census Income** dataset (UCI / Kaggle mirror). Do not use Titanic or Student Performance as the primary dataset — see prior discussion on why.
- Write `config.py` — one file holding every tunable constant: severities `[0.1, 0.2, 0.3, 0.4, 0.5]`, seeds `[0, 1, 2, 3, 4]`, model list, dataset path. Every other module reads from here — nothing hardcoded elsewhere.

**Day 2 — EDA + preprocessing**
- Basic EDA: nulls, dtypes, class balance, feature ranges (you need these ranges later to scale noise/outlier severity meaningfully).
- Build `data/preprocessing.py`: encode categoricals, scale numerics, train/test split (stratified, fixed seed).
- Save a clean `train.csv` / `test.csv` so every later step starts from the same frozen split.

**Done when:** you can load, preprocess, and split the dataset in one function call, and you have a written note of each feature's min/max/std (you'll need this to sanity-check perturbation severity later).

---

## Phase 2: Baseline Models (Day 3)

- Train 3 models on the **clean** training data: Logistic Regression, Random Forest, Gradient Boosting.
- Decision Tree is deliberately dropped — it's the same family as Random Forest and adds no new failure mode to compare.
- Record baseline metrics (accuracy, precision, recall, F1) on the clean test set. This is your `severity = 0` anchor point — every later robustness number is relative to this.
- Save trained models (`joblib`) so the sweep in Phase 3 doesn't retrain from scratch every run.

**Done when:** you have one baseline metrics table, one row per model, saved to disk.

---

## Phase 3: Stress-Test Engine (Day 4–6)

This is the core of the project — take the extra days here, it's worth it.

**Day 4 — Noise + missing values**
```python
def add_noise(X, severity, seed):
    rng = np.random.default_rng(seed)
    std = X.std(axis=0) * severity
    return X + rng.normal(0, std, X.shape)

def introduce_missing(X, severity, seed, strategy="median"):
    rng = np.random.default_rng(seed)
    mask = rng.random(X.shape) < severity
    X_missing = X.copy()
    X_missing[mask] = np.nan
    return X_missing.fillna(X_missing.agg(strategy))
```
Document your imputation `strategy` choice explicitly — it's part of the experiment, not an implementation detail to skip over.

**Day 5 — Outliers + covariate shift**
```python
def add_outliers(X, severity, seed, scale=5):
    rng = np.random.default_rng(seed)
    X_out = X.copy()
    n_rows = int(len(X) * severity)
    idx = rng.choice(len(X), n_rows, replace=False)
    X_out.iloc[idx] = X_out.iloc[idx] * scale
    return X_out

def shift_distribution(X, severity, seed, feature):
    # scoped definition: covariate shift only — shifts one feature's mean
    X_shift = X.copy()
    X_shift[feature] += X[feature].std() * severity
    return X_shift
```
Write one sentence in your notes right now: *"shift_distribution simulates covariate shift by shifting a feature's mean; it does not simulate label or concept shift."* You'll reuse this sentence in your report and viva.

**Day 6 — The sweep runner**
This is what actually makes the project statistically valid — build `stress_tests/sweep.py`:
```python
results = []
for model_name, model in models.items():
    for perturbation_name, perturb_fn in perturbations.items():
        for severity in config.SEVERITIES:
            for seed in config.SEEDS:
                X_test_perturbed = perturb_fn(X_test, severity, seed)
                preds = model.predict(X_test_perturbed)
                metrics = compute_metrics(y_test, preds)
                results.append({
                    "model": model_name, "perturbation": perturbation_name,
                    "severity": severity, "seed": seed, **metrics
                })
```
That's 3 models × 4 perturbations × 5 severities × 5 seeds = **300 evaluation runs**, saved to one `results.csv`. This single file is what every later phase reads from.

**Done when:** `results.csv` exists with ~300 rows and no errors, and you've spot-checked a few rows by hand to confirm the severity actually changed the data as expected.

---

## Phase 4: Evaluation, Aggregation, Robustness Index (Day 7–8)

**Day 7 — Aggregation**
- Group `results.csv` by `(model, perturbation, severity)` and compute mean ± std across the 5 seeds.
- This is what turns "one run said X" into "across 5 runs, X held with this much variance."

**Day 8 — Robustness index**
```python
retention = metric_at_severity / metric_at_severity_zero
robustness_index = mean(retention across all tested severities)
```
- Compute one robustness index per model per perturbation type, then average across perturbation types for one overall score per model.
- Optional but strong addition: run a paired comparison (e.g. paired t-test across seeds) between your top two models' robustness indices — gives you a real statistical claim instead of "Model A's number is bigger."

**Done when:** you have a table — 3 models × 1 robustness index each — that you could screenshot and defend line by line.

---

## Phase 5: Visualization & Insight Generator (Day 9–10)

**Day 9 — Visualization**
- Primary chart: **degradation curve** — x-axis severity, y-axis metric, one line per model, one chart per perturbation type. This is the chart that actually communicates the finding; bar charts of before/after hide the shape of the decline.
- Secondary: a heatmap of robustness index (model × perturbation type).

**Day 10 — Insight generator**
Not hardcoded strings — thresholds evaluated against your real numbers:
```python
if robustness_index[model] < 0.7:
    print(f"{model} degrades significantly under stress (index={robustness_index[model]:.2f})")

best = max(robustness_index, key=robustness_index.get)
worst = min(robustness_index, key=robustness_index.get)
print(f"{best} is the most robust model overall (index={robustness_index[best]:.2f}); "
      f"{worst} is the least robust (index={robustness_index[worst]:.2f})")
```
If a statement can't trace back to a number in your results table, cut it.

**Done when:** every printed insight has a number attached, and you could point to the exact line of `results.csv` that produced it.

---

## Phase 6: Custom Web App — FastAPI + React (Day 11–15)

The web app is a **presentation layer only** — it does not reimplement any ML
logic. `main.py` (from Phases 1–5) already produces `results.csv`; the backend
just serves it and the frontend just renders it. If you find yourself
recomputing metrics inside a route handler, stop — that logic belongs in
`evaluation/`, not `backend/`.

**Day 11 — Backend API**
- Build `backend/app.py` (FastAPI). Read-only endpoints:
  - `GET /api/models` — model list + baseline metrics
  - `GET /api/perturbations` — perturbation types + scope text (e.g. the
    covariate-shift-only disclaimer)
  - `GET /api/results` — aggregated mean ± std per (model, perturbation, severity)
  - `GET /api/robustness-index` — one score per model
  - `GET /api/insights` — generated insight strings
- Test every endpoint with `curl` or the FastAPI `/docs` page before touching the frontend.

**Day 12 — Frontend scaffold**
- Set up the React app (`frontend/`), basic layout: model selector, perturbation selector, results table.
- Wire it to the backend — confirm real data renders before adding any styling.

**Day 13 — Charts**
- Degradation curve (recharts line chart): x = severity, y = metric, one line per model, one chart per perturbation type. This is still the chart that carries the project.
- Robustness index displayed as a small table or card set, not buried in a tooltip.

**Day 14 — Severity interactivity**
- Slider snapped to the 5 tested severities (`0.1`–`0.5`) — this queries real, seed-averaged data from `/api/results`, so what the user sees on drag is never fabricated.
- Optional: `/api/live-perturb` for a severity outside the tested grid, computed on the fly from a single seed. If you build this, the UI must visibly label it as a single-seed preview — don't let it look like the same statistically averaged number as the rest of the dashboard.
- Insight text updates alongside the chart when the model/perturbation selection changes.

**Day 15 — Polish pass**
- Loading states, empty states, basic responsive layout.
- One clear "how to read this chart" caption near the degradation curve — reviewers shouldn't have to guess what retention vs. severity means.

**Done when:** you can open the app, pick a model, drag the severity slider across the tested grid, and watch the chart, robustness index, and insight text update — and every number on screen traces back to a row in `results.csv`.

---

## Phase 7: Testing, Documentation, Polish (Day 16–17)

**Day 16 — Testing**
- Basic unit tests for each stress-test function: does `severity=0` return the original data unchanged? Does `severity=1` behave sanely at the extreme? Does the seed make results reproducible?
- Sanity check: does robustness index ever exceed 1 (it shouldn't, unless something's wrong upstream)?

**Day 17 — Documentation**
- `README.md`: problem statement, architecture diagram, how to run, how to interpret the robustness index.
- A short methodology section stating your scope explicitly: Track A only (train clean, test corrupted — measures generalization to real-world corruption, not robustness-via-augmentation), covariate shift only (not label/concept shift). Say this proactively — it pre-empts the exact questions a reviewer would otherwise ask.
- A one-paragraph limitations section (small model set, tabular-only, simplified shift definition). Naming your own limitations is what makes the rest of the claims more credible, not less.

**Done when:** someone who has never seen the project could clone the repo, read the README, and reproduce your main result.

---

## Stretch (Day 18–19, optional but you said you have the time)

Pick from this list based on what you want the project to signal:

- **Second dataset** (e.g. a credit-scoring dataset) to show the framework generalizes beyond one dataset — strong if you want to emphasize "framework," not "one experiment."
- **Deploy the dashboard** (Streamlit Community Cloud) so you have a live link for your resume/portfolio instead of just a repo.
- **Track B experiment**: train on augmented (perturbed) data and compare robustness — a genuinely different, second experiment, keep it clearly separated from Track A in your writeup.
- **Presentation prep**: build the PPT (architecture diagram, degradation curve as the hero visual, robustness index table, 2–3 sentence resume line), and write out likely viva questions with answers — "why average retention instead of raw accuracy drop," "why only covariate shift," "why these 3 models."

---

## Folder structure (reference)

```
project/
│── data/
│   ├── raw/
│   └── preprocessing.py
│── models/
│   └── train.py
│── stress_tests/
│   ├── perturbations.py
│   └── sweep.py
│── evaluation/
│   ├── aggregate.py
│   └── robustness_index.py
│── visualization/
│   └── plots.py
│── insights/
│   └── generator.py
│── config.py
│── main.py                # runs the offline pipeline end-to-end → results.csv
│── backend/
│   ├── app.py              # FastAPI entrypoint
│   ├── routes/
│   └── schemas.py
│── frontend/
│   ├── src/                 # React app: selectors, degradation chart, severity slider
│   ├── package.json
│   └── ...
│── tests/
│── requirements.txt
│── README.md
```
