# Predicting Pharmaceutical Production Yield Using Machine Learning

**A Comparative Study of Regression and Regularization Techniques** — CS 5821 Machine Learning project

## Purpose and hypothesis

**Research question:** How accurately can pharmaceutical production yield be predicted from historical
manufacturing measurements, and which machine-learning method offers the best balance between
prediction accuracy, complexity and interpretability?

**Hypothesis:** With 57 correlated predictors and only 176 batches, ordinary least squares will
overfit. Methods that control complexity (Ridge, Lasso, PCR) will generalise better, and the sparse
Lasso will give the best accuracy-interpretability trade-off.

This is a retrospective, academic statistical analysis of a public teaching dataset. Its findings
are predictive associations, not causal effects, and they are not recommendations for operating
a real manufacturing process.

## Headline results (random 80/20 split, `random_state=42`)

| Model | CV RMSE | Test MSE | Test RMSE | Test MAE | Test R² | Selected hyperparameters |
|---|---|---|---|---|---|---|
| Linear Regression | 10.215 | 2.073 | 1.440 | 1.109 | 0.126 | none |
| Ridge | 1.582 | 1.544 | 1.243 | 1.023 | 0.349 | α = 125.9 |
| **Lasso** | **1.195** | **1.187** | **1.089** | **0.859** | **0.500** | α = 0.0708 (22 of 56 features kept) |
| PCR | 1.352 | 1.664 | 1.290 | 1.022 | 0.298 | 22 components |
| KNN Regression | 1.327 | 1.722 | 1.312 | 1.130 | 0.274 | k = 12, distance weights |

The model was selected by CV RMSE on the training data only; the test set was used once at the end.
Yield has SD 1.85, so always predicting the mean would give an RMSE of about 1.85.
See [reports/findings.md](reports/findings.md) for the full interpretation.

## Dataset

`ChemicalManufacturingProcess`, 176 batches × 58 columns: `Yield` (target, %), 12
`BiologicalMaterial` and 45 `ManufacturingProcess` measurements; 106 missing predictor values.

* Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.
* R package *AppliedPredictiveModeling* — [documentation](https://search.r-project.org/CRAN/refmans/AppliedPredictiveModeling/html/ChemicalManufacturingProcess.html)
* CSV mirror used here: <https://raw.githubusercontent.com/rwepa/DataDemo/refs/heads/master/ChemicalManufacturingProcess.csv>

A copy is included at `data/ChemicalManufacturingProcess.csv`. If it is missing, the code downloads it
automatically. Offline, you can create the file in R and save it at that path:

```r
library(AppliedPredictiveModeling); data(ChemicalManufacturingProcess)
write.csv(ChemicalManufacturingProcess, "data/ChemicalManufacturingProcess.csv", row.names = FALSE)
```

## Installation

Python 3.9 or newer is required. From the repository root (`YieldSense/`):

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

On Windows, run `.venv\Scripts\activate` instead.

```bash
pip install -r requirements.txt
```

## Running

Run the full analysis. It writes every figure to `reports/figures/` and every table to `reports/tables/`,
and takes about 2 minutes because of the 20-split robustness check:

```bash
python -m src.train_models
```

Run the notebook interactively:

```bash
jupyter notebook notebooks/pharma_yield_analysis.ipynb
```

Or execute it headlessly:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/pharma_yield_analysis.ipynb
```

Run the tests:

```bash
python -m pytest -q
```

### Reproducibility

All randomness is seeded: the train/test split, the CV folds (`KFold(5, shuffle=True, random_state=42)`),
`GroupShuffleSplit`, and the 20 repeated splits (seeds 0–19). Re-running either the script or the
notebook reproduces the numbers above exactly with the tested library versions.

## Methodology

**Preprocessing** (inside every scikit-learn `Pipeline`, so it is refit in each CV fold and never sees test data):
1. *Near-zero-variance filter* (Kuhn & Johnson rule) — removes `BiologicalMaterial07` (173/176 identical values).
2. *Median imputation* — robust to the outliers found in EDA.
3. *StandardScaler* — needed by Ridge, Lasso, PCA and KNN.

**Models**
| Model | Idea | Tuned by 5-fold CV |
|---|---|---|
| Multiple linear regression | least squares on all 56 predictors (baseline) | – |
| Ridge | least squares + α Σβ² (shrinks all coefficients) | α ∈ [10⁻², 10⁴], 61 values |
| Lasso | least squares + α Σ\|β\| (sets some coefficients exactly to 0) | α ∈ [10⁻³, 1], 61 values |
| PCR | PCA → linear regression on the first M components | M = 1…40 |
| KNN regression | average yield of the k most similar batches | k = 1…30, uniform/distance weights |

A statsmodels OLS summary is also produced for an interpretable 8-variable subset. The subset is chosen
from the training data by correlation with yield, excluding predictors with |r| > 0.7 to any already chosen.

**Metrics:** MSE, RMSE, MAE and R² on the 36-batch test set, plus 5-fold CV RMSE on training data.

**Batch dependency:** only 108 distinct biological-material profiles exist among 176 batches,
so 103 batches share raw material with another batch. Two sensitivity analyses address this:
GroupKFold vs KFold on the training data, and a fully grouped 80/20 re-run. A third analysis
repeats the full pipeline over 20 random splits to measure how noisy a single 36-row test set is.

## File structure

```
YieldSense/
├── data/ChemicalManufacturingProcess.csv   # real dataset (downloaded)
├── notebooks/pharma_yield_analysis.ipynb   # executed, narrated analysis
├── src/
│   ├── data_loader.py      # download/load/validate, batch groups
│   ├── preprocessing.py    # split, NZV filter, imputer+scaler pipeline, EDA helpers
│   ├── train_models.py     # 5 model pipelines, GridSearchCV, sensitivity analyses; entry point
│   ├── evaluate_models.py  # metrics, comparison table, coefficients, statsmodels OLS, full run
│   └── visualizations.py   # all figures
├── reports/
│   ├── figures/            # 15 PNGs (see findings.md for the list)
│   ├── tables/             # CSV/TXT result tables
│   └── findings.md         # verified findings + report/presentation outline
├── tests/test_pipeline.py  # pytest suite (12 tests)
├── requirements.txt
└── README.md
```

## Limitations

* Small sample (176 batches, 36 test batches): test RMSE varies by about ±0.3 between random splits.
* Batches sharing raw material are not independent, so random-split accuracy is somewhat optimistic
  (grouped CV RMSE is about 10 % higher for Lasso).
* Strong multicollinearity (14 predictor pairs with |r| ≥ 0.9) makes individual coefficients unstable.
* Median imputation ignores imputation uncertainty. A few process columns contain isolated zeros that
  look like recording artefacts; they were kept as recorded.
* The data are observational, so the associations are not causal.
