# Robustness Evaluation Framework for ML Models

Train ML models, stress-test them under parameterized data perturbations
(noise, missing values, outliers, covariate shift), and produce a
**statistically averaged robustness index** per model — not a single
before/after snapshot.

Instead of asking *"how accurate is this model on clean data?"*, this project
asks *"how much of its performance does it retain as the input data degrades?"*
— and answers it by averaging over 5 random seeds so the number reflects a
distribution, not one lucky (or unlucky) run.

## Problem statement

A model that wins on a clean test set is not necessarily the model you want in
production, where data arrives noisy, incomplete, and shifted from the training
distribution. This framework quantifies **generalization to real-world
corruption** by evaluating each model across a grid of perturbation types and
severities, then collapsing the results into one comparable robustness score
per model.

Dataset: **Adult Census Income** (UCI) — mixed categorical/numeric features
with real class imbalance (~76% ≤50K / 24% >50K).

## Headline result

On this dataset, the model with the highest clean-data accuracy is **not** the
most robust one:

| model | clean accuracy (severity 0) | overall robustness index |
|---|---|---|
| gradient_boosting | **0.869** | 0.985 |
| random_forest | 0.861 | **0.987** |
| logistic_regression | 0.856 | 0.985 |

Gradient Boosting leads on clean accuracy but **Random Forest is the most
robust** — its accuracy decays more gently as perturbation severity rises. A
paired t-test across seeds (Random Forest vs. Gradient Boosting retention)
gives t = 6.10, p = 0.0037, so the robustness gap is statistically
significant, not noise. This is exactly the kind of finding a single
clean-accuracy number hides.

*(Numbers regenerate from `results.csv`; rerun the pipeline to reproduce.)*

## Architecture

The offline pipeline is the single source of truth. The web app is a
presentation layer only — it never recomputes an ML metric.

```
                        OFFLINE PIPELINE  (python main.py)
  ┌──────────────────────────────────────────────────────────────────┐
  │  data/preprocessing.py   clean → encode → scale → stratified split │
  │            │                                                        │
  │            ▼                                                        │
  │  models/train.py         train LR / RF / GB on CLEAN data          │
  │            │              → baseline_metrics.csv (severity=0 anchor)│
  │            ▼                                                        │
  │  stress_tests/sweep.py   3 models × 4 perturbations × 5 severities  │
  │            │              × 5 seeds = 300 runs                      │
  │            ▼                                                        │
  │        results.csv       ◀── SINGLE SOURCE OF TRUTH                 │
  └──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼───────────────────────┐
        ▼                     ▼                         ▼
  evaluation/           visualization/            insights/
  aggregate.py          plots.py                  generator.py
  robustness_index.py   (degradation curves,      (threshold-checked
  (mean±std, index)      heatmap)                   statements)
        │
        ▼
  ┌───────────────────────────┐     ┌──────────────────────────────┐
  │ backend/  FastAPI          │◀────│ frontend/  React + recharts   │
  │ read-only endpoints over   │ /api│ selectors, degradation chart, │
  │ results.csv                │────▶│ severity slider, insight text │
  └───────────────────────────┘     └──────────────────────────────┘
```

## How to run

Prerequisites: Python 3.11+ and Node.js 18+.

### 1. Offline pipeline (produces `results.csv`)

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

python -m data.download          # fetch the Adult dataset into data/raw/
python main.py                   # preprocessing → training → 300-run sweep → results.csv
```

`python main.py` regenerates everything downstream: `data/train.csv`,
`data/test.csv`, the fitted models, `models/baseline_metrics.csv`, and
`results.csv`. Optionally run the visualizers and insight generator:

```bash
python -m visualization.plots    # degradation curves + heatmap → visualization/plots/
python -m insights.generator     # printed, threshold-checked insight statements
```

### 2. Tests

```bash
pytest tests/
```

### 3. Web app

```bash
# terminal 1 — backend
uvicorn backend.app:app --reload         # http://127.0.0.1:8000  (docs at /docs)

# terminal 2 — frontend
cd frontend && npm install && npm run dev # http://localhost:5173
```

The frontend proxies `/api` to the backend, so start both.

## How to interpret the robustness index

For a given metric (default: accuracy):

```
retention(severity)  = metric(severity) / metric(severity = 0)
robustness_index     = mean( retention across the tested severities )
overall_index(model) = mean( robustness_index across perturbation types )
```

- `retention = 1.0` → the perturbation cost the model nothing at that severity.
- `retention = 0.9` → the model kept 90% of its clean-data metric.
- A **higher overall index = more robust.** The index is bounded at ≈1 from
  above; a value meaningfully above 1 would signal an upstream bug (a
  perturbation "improving" the model), which the test suite guards against.

Because every number is anchored to that model's *own* clean baseline, the
index measures **relative degradation**, so models with different clean
accuracies are still directly comparable on robustness.

## Methodology & scope

State the scope up front — it pre-empts the obvious reviewer questions.

- **Track A only** (train on clean data, evaluate on perturbed data). This
  measures generalization to real-world corruption. It is *not*
  robustness-via-augmentation (Track B: training on perturbed data), which
  would be a separate, clearly labeled experiment.
- **Covariate shift only.** `shift_distribution` shifts one feature's mean by
  `severity × std`. It does **not** simulate label shift or concept shift — the
  feature↔label relationship is left intact.
- **Imputation strategy is part of the experiment.** `introduce_missing` nulls
  numeric values at rate `severity`, then imputes with the **median**. The
  choice of imputation determines how much signal survives missingness, so it
  is a documented design decision, not an implementation detail.
- **Perturbations touch numeric features only.** Adding Gaussian noise or
  scaled outliers to a one-hot categorical indicator would produce values that
  correspond to no real category; the four perturbations are physically
  meaningful only on continuous measurements.
- **Statistical averaging.** Every (model × perturbation × severity) cell is
  the mean over 5 seeds, reported with its std — one noisy run is never used as
  a result.

## Limitations

Naming the limits is what makes the rest of the claims credible.

- **Small model set** (3 models) and **tabular-only** — no deep/sequence/vision
  models, no image or text corruption.
- **Simplified shift definition** — covariate shift on a single feature's mean,
  not multivariate or label/concept shift.
- **One dataset.** The framework generalizes, but the specific numbers are
  Adult-Census-Income specific. A second dataset is a natural extension.
- **Fixed perturbation magnitudes** (e.g. outlier scale ×5) chosen to produce a
  visible-but-not-catastrophic decline on this dataset.

## Folder structure

```
data/            download, EDA, preprocessing → frozen train/test split
models/          train.py → fitted models + baseline_metrics.csv
stress_tests/    perturbations.py (fn(X, severity, seed)) + sweep.py
evaluation/      aggregate.py (mean±std) + robustness_index.py
visualization/   plots.py (degradation curves, heatmap)
insights/        generator.py (threshold-checked statements)
config.py        every severity / seed / model / path constant
main.py          offline pipeline entrypoint → results.csv
backend/         FastAPI read-only API over results.csv
frontend/        React + recharts dashboard
tests/           pytest suite (perturbation invariants, index sanity)
```

## Tech stack

Python · pandas · numpy · scikit-learn · scipy · matplotlib · seaborn · joblib ·
FastAPI · uvicorn · React · Vite · recharts · pytest
