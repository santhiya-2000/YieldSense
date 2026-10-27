# Findings — Predicting Pharmaceutical Production Yield

All numbers below come from `python -m src.train_models` / the executed notebook
(random 80/20 split with `random_state=42`; 140 training and 36 test batches). Tables are in
`reports/tables/` and figures in `reports/figures/`.

## 1. Data facts (verified)

* 176 batches, 57 predictors (12 biological-material, 45 manufacturing-process), target `Yield`.
* Yield: mean 40.18 %, SD 1.85, range 35.25–46.34, slight right skew.
* 106 missing predictor values in 28 columns and 24 batches; the worst column is `ManufacturingProcess03` (15 missing, 8.5 %).
* 14 predictor pairs have |r| ≥ 0.90 (training set); 36 of 56 predictors have VIF > 10 in the full OLS model, and 3 are exactly collinear (VIF = ∞).
* `BiologicalMaterial07` is near-constant (173/176 identical values) and was removed.
* A few batches are more than 10 SD from the median on one process variable (zeros in MP16/18/26/43).
* **Batch dependency:** only 108 distinct biological-material profiles exist; 33 occur three times. Within-profile yield SD is 0.49 vs 1.85 overall. With the random split, 18 of 36 test batches share a raw-material profile with a training batch.

## 2. Model comparison (`tables/model_comparison.csv`)

| Model | CV RMSE (±SD) | Train RMSE | Test MSE | Test RMSE | Test MAE | Test R² | Hyperparameters |
|---|---|---|---|---|---|---|---|
| Linear Regression | 10.215 (±10.63) | 0.810 | 2.073 | 1.440 | 1.109 | 0.126 | — |
| Ridge | 1.582 (±0.54) | 1.068 | 1.544 | 1.243 | 1.023 | 0.349 | α = 125.9 |
| **Lasso** | **1.195 (±0.34)** | 1.010 | **1.187** | **1.089** | **0.859** | **0.500** | α = 0.0708 |
| PCR | 1.352 (±0.26) | 1.042 | 1.664 | 1.290 | 1.022 | 0.298 | 22 PCs |
| KNN Regression | 1.327 (±0.34) | 0.000 | 1.722 | 1.312 | 1.130 | 0.274 | k = 12, distance |

Lasso test residuals show no funnel shape and are close to normal (Q–Q plot), but they are shifted
upward: the mean residual is +0.52 and 26 of 36 are positive. The test batches have a higher mean
yield (40.62) than the training batches (40.06), and the regularised model is pulled toward the
training mean, so high-yield batches are under-predicted.

## 3. Answers to the research questions

**Which model achieved the lowest prediction error?** Lasso: lowest CV RMSE (1.195), which is how it
was selected, and best on every test metric (RMSE 1.089, MAE 0.859, R² 0.500). It explains about half
of the batch-to-batch yield variation on unseen batches.

**Did regularisation improve generalisation?** Yes, strongly. OLS CV RMSE was 10.2 because two
folds exceeded 20 when extreme batches were held out; Ridge reduced it to 1.58 and Lasso to 1.20.
Test RMSE fell from 1.44 (OLS) to 1.24 (Ridge) and 1.09 (Lasso).

**Did PCA reduce dimensionality without much loss of accuracy?** PCR reduced 56 correlated predictors to
22 uncorrelated components (92.9 % of the variance; the first PC alone explains 17.9 %). It improved
greatly on OLS (test RMSE 1.29 vs 1.44), but it was less accurate than Lasso because the components
are built without looking at yield.

**Which features are most strongly associated with yield?** The models agree on these
(standardised Lasso coefficient in brackets): ManufacturingProcess32 (+0.76), ManufacturingProcess09
(+0.55), ManufacturingProcess17 (−0.30), BiologicalMaterial06 (+0.27), ManufacturingProcess37 (−0.17),
ManufacturingProcess39 (+0.14). Ridge gives the same signs. In the 8-variable statsmodels OLS
(adj. R² 0.59, all VIF < 3), MP32 (+0.94, p < 0.001), MP13 (−0.55, p = 0.002), MP11 (+0.38, p = 0.010)
and BM06 (+0.35, p = 0.035) are significant. These are predictive associations, not causal effects.
The subset p-values are optimistic because the variables were selected on the same data.

**Was there evidence of overfitting?** Yes. OLS has training RMSE 0.81 but CV RMSE 10.2 and the worst
test R² (0.13). KNN with k = 1 or distance weighting reproduces its training data exactly
(training RMSE 0). Ridge, Lasso and PCR show modest train/CV gaps (CV/train ratio 1.2–1.5).

**Why is Lasso preferable?** It has the best accuracy and is also the simplest model: it uses 22 of
56 predictors, its coefficients are readable, and it handles collinearity by keeping one member of
each near-duplicate group. KNN is slightly more stable across splits, but less accurate and not
interpretable.

## 4. Robustness checks

* **Grouped CV (`tables/grouped_cv_comparison.csv`)**: keeping raw-material profiles intact raises CV RMSE
  by 10.5 % (Lasso), 10.2 % (PCR), 5.0 % (KNN) and 32 % (OLS); Ridge −1.4 %. Lasso stays best (1.32).
* **Grouped 80/20 hold-out (`tables/grouped_holdout_evaluation.csv`, 146/30 batches)**: Lasso is best again
  (test RMSE 0.936, R² 0.725). This test set happened to be easier and contained none of the extreme
  batches, so OLS scored 0.971 there.
* **20 repeated random splits (`tables/repeated_split_summary.csv`)**: mean test RMSE was Lasso 1.281,
  KNN 1.346, PCR 1.435, Ridge 1.555 and OLS 4.665 (max 28.2). Lasso was best in 9 of 20 splits and KNN
  in 5. A single test RMSE varies by about ±0.3, so only "Lasso first, OLS last" is robust.

## 5. Limitations

1. Small sample: 176 batches and 36 test rows give noisy estimates.
2. Non-independent batches (shared raw material) make random-split accuracy somewhat optimistic.
3. Multicollinearity makes individual coefficients unstable; Lasso's choice within a correlated group is partly arbitrary.
4. Median imputation ignores uncertainty; suspected placeholder zeros were left unchanged.
5. The data are observational and come from a single product and site, so results do not support
   causal claims or process-setting changes.

## 6. Figures for the report and slides

| File | Use |
|---|---|
| `01_yield_distribution.png` | Target distribution |
| `02_missing_values.png` | Missingness |
| `03_predictor_histograms.png` | Predictor distributions |
| `04_correlation_heatmap.png` | Correlations / multicollinearity |
| `05_predictor_boxplots.png` | Outliers |
| `06_pairwise_correlation_distribution.png` | Multicollinearity overview |
| `07_cv_tuning_curves.png` | Hyperparameter tuning, bias–variance trade-off |
| `08_pca_cumulative_variance.png` | PCA explained variance |
| `09_model_comparison_rmse.png` | **Main result** |
| `10_actual_vs_predicted.png` | Selected model fit |
| `11_residual_plot.png` | Selected model residuals |
| `12_ridge_vs_lasso_coefficients.png` | Regularisation and feature associations |
| `13_ols_residual_diagnostics.png` | OLS assumptions |
| `14_grouped_validation.png` | Batch-dependency sensitivity |
| `15_repeated_split_rmse.png` | Robustness across splits |

## 7. Suggested report outline (~5 pages)

1. **Introduction** (½ p): Pharma 4.0 context, research question, hypothesis.
2. **Data** (½ p): source and citation, variables, missingness, batch structure (figures 01, 02).
3. **Exploratory analysis** (¾ p): skew, outliers, multicollinearity (figures 04, 06; optionally 05).
4. **Methods** (1 p): leakage-safe pipeline, 80/20 split, 5-fold CV, the five models, metrics.
5. **Results** (1¼ p): comparison table, figure 09, tuning curves (07), actual vs predicted (10), Ridge vs Lasso (12), PCA (08).
6. **Robustness and discussion** (¾ p): grouped validation (14), repeated splits (15), overfitting, best-model justification.
7. **Conclusion and limitations** (¼ p), then **References**.

## 8. Suggested 10-minute presentation (≈ 10 slides)

1. Title and research question · 2. Why yield prediction matters (non-causal framing) · 3. Dataset and EDA (01, 04)
· 4. Challenges: p ≈ n/2.5, collinearity, shared batches · 5. Pipeline and leakage control · 6. The five models
· 7. Tuning curves (07) · 8. Results table and figure 09 · 9. Lasso: actual vs predicted (10) and coefficients (12)
· 10. Robustness (14, 15), limitations and conclusion.

## References

1. Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*. Springer.
2. Kuhn, M. & Johnson, K. (2018). *AppliedPredictiveModeling* R package, v1.1-7. CRAN.
3. James, G., Witten, D., Hastie, T. & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
4. Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
5. Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1), 267–288.
6. Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression. *Technometrics*, 12(1), 55–67.
7. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*, 12, 2825–2830.
