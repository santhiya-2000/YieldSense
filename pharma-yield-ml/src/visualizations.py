"""Presentation-quality figures. Every function saves a PNG to reports/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; notebooks still display returned figures

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from src.data_loader import FIGURES_DIR

# Palette (validated categorical order + neutral inks)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIVERGING = sns.blend_palette(["#2a78d6", "#f0efec", "#e34948"], as_cmap=True)
DPI = 200


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
        "axes.titleweight": "bold", "axes.titlesize": 14, "axes.titlelocation": "left",
        "legend.frameon": False, "savefig.bbox": "tight",
    })


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=DPI)
    return fig


# ------------------------------------------------------------------- EDA plots

def plot_yield_distribution(y: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [4, 1]})
    sns.histplot(y, bins=20, kde=True, color=BLUE, edgecolor="white", ax=ax1)
    ax1.axvline(y.mean(), color=ORANGE, lw=2, ls="--", label=f"Mean = {y.mean():.2f}")
    ax1.axvline(y.median(), color=AQUA, lw=2, ls=":", label=f"Median = {y.median():.2f}")
    ax1.set(title=f"Production yield distribution (n = {len(y)})", ylabel="Number of batches")
    ax1.legend()
    sns.boxplot(x=y, color=BLUE, width=0.5, ax=ax2, flierprops={"markersize": 6})
    ax2.set(xlabel="Yield (%)")
    skew = stats.skew(y)
    ax1.text(0.99, 0.95, f"SD = {y.std():.2f}\nSkewness = {skew:.2f}", transform=ax1.transAxes,
             ha="right", va="top", color=INK2, fontsize=11)
    return _save(fig, "01_yield_distribution.png")


def plot_missing_values(missing: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 7))
    m = missing.sort_values("percent")
    ax.barh(m.index, m["percent"], color=BLUE, height=0.7)
    for i, (cnt, pct) in enumerate(zip(m["missing"], m["percent"])):
        ax.text(pct + 0.1, i, f"{cnt}", va="center", fontsize=9, color=INK2)
    ax.set(title=f"Missing values by predictor ({len(m)} of 57 predictors affected)",
           xlabel="Missing (% of 176 batches)", ylabel="")
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", visible=False)
    return _save(fig, "02_missing_values.png")


def plot_predictor_histograms(X: pd.DataFrame, features: list[str]):
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.4 * rows))
    for ax, f in zip(axes.flat, features):
        sns.histplot(X[f].dropna(), bins=20, color=BLUE, edgecolor="white", ax=ax)
        ax.set_title(f.replace("ManufacturingProcess", "MP").replace("BiologicalMaterial", "BM"),
                     fontsize=12)
        ax.set(xlabel="", ylabel="Count")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle("Distributions of the predictors most correlated with yield (training set)",
                 x=0.01, ha="left", fontweight="bold", fontsize=14)
    fig.tight_layout()
    return _save(fig, "03_predictor_histograms.png")


def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series, features: list[str]):
    data = X[features].assign(Yield=y)
    data.columns = [c.replace("ManufacturingProcess", "MP").replace("BiologicalMaterial", "BM")
                    for c in data.columns]
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap=DIVERGING, vmin=-1, vmax=1, center=0, annot=True,
                fmt=".2f", annot_kws={"size": 8}, linewidths=1, linecolor="white",
                cbar_kws={"label": "Pearson r", "shrink": 0.8}, square=True, ax=ax)
    ax.set_title("Correlation of yield with its 12 most-correlated predictors (training set)")
    ax.grid(False)
    ax.tick_params(axis="x", rotation=45, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    return _save(fig, "04_correlation_heatmap.png")


def plot_predictor_boxplots(X: pd.DataFrame, features: list[str]):
    z = (X[features] - X[features].mean()) / X[features].std()
    z.columns = [c.replace("ManufacturingProcess", "MP").replace("BiologicalMaterial", "BM")
                 for c in z.columns]
    long = z.melt(var_name="Predictor", value_name="z")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.boxplot(data=long, x="Predictor", y="z", color=BLUE, width=0.6,
                flierprops={"markerfacecolor": ORANGE, "markeredgecolor": ORANGE, "markersize": 5},
                ax=ax)
    ax.axhline(3, color=MUTED, ls="--", lw=1)
    ax.axhline(-3, color=MUTED, ls="--", lw=1)
    ax.set(title="Outlier check: standardised predictors (dashed lines = ±3 SD)",
           xlabel="", ylabel="Standardised value (z-score)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "05_predictor_boxplots.png")


def plot_pairwise_correlation_hist(X: pd.DataFrame):
    corr = X.corr().abs().values
    vals = corr[np.triu_indices_from(corr, k=1)]
    vals = vals[~np.isnan(vals)]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(vals, bins=40, color=BLUE, edgecolor="white", ax=ax)
    for t, c in [(0.75, YELLOW), (0.9, ORANGE)]:
        ax.axvline(t, color=c, lw=2, ls="--", label=f"|r| ≥ {t}: {(vals >= t).sum()} pairs")
    ax.set(title=f"Multicollinearity: |r| for all {len(vals)} predictor pairs (training set)",
           xlabel="Absolute Pearson correlation |r|", ylabel="Number of pairs")
    ax.legend()
    return _save(fig, "06_pairwise_correlation_distribution.png")


# ------------------------------------------------------------- modelling plots

def plot_cv_curves(searches: dict):
    """Validation curves: CV RMSE against each model's tuning parameter."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    specs = [("Ridge", "param_model__alpha", "Regularisation strength α (log scale)", True),
             ("Lasso", "param_model__alpha", "Regularisation strength α (log scale)", True),
             ("PCR", "param_pca__n_components", "Number of principal components", False),
             ("KNN Regression", "param_model__n_neighbors", "Number of neighbours k", False)]
    for ax, (name, param, xlabel, logx) in zip(axes.flat, specs):
        s = searches[name]
        res = pd.DataFrame(s.cv_results_)
        if name == "KNN Regression":
            for w, color in [("uniform", BLUE), ("distance", ORANGE)]:
                r = res[res["param_model__weights"] == w]
                ax.plot(r[param].astype(float), -r["mean_test_score"], color=color, lw=2,
                        marker="o", ms=4, label=f"CV RMSE ({w} weights)")
            r = res[res["param_model__weights"] == "uniform"]
            ax.plot(r[param].astype(float), -r["mean_train_score"], color=MUTED, lw=1.5,
                    ls="--", label="Training RMSE (uniform)")
        else:
            x = res[param].astype(float)
            mean, sd = -res["mean_test_score"], res["std_test_score"]
            ax.plot(x, mean, color=BLUE, lw=2, label="CV RMSE")
            ax.fill_between(x, mean - sd, mean + sd, color=BLUE, alpha=0.12, label="±1 SD across folds")
            ax.plot(x, -res["mean_train_score"], color=MUTED, lw=1.5, ls="--", label="Training RMSE")
        best = float(s.best_params_[param.replace("param_", "")])
        ax.axvline(best, color=ORANGE, ls=":", lw=2, label=f"Selected = {best:.3g}")
        ax.set(title=f"{name}: lowest CV RMSE at {xlabel.split(' (')[0].lower()} = {best:.3g}",
               xlabel=xlabel, ylabel="RMSE (yield %)")
        if logx:
            ax.set_xscale("log")
        ax.legend(fontsize=9)
    if "KNN Regression" in searches:
        p = searches["KNN Regression"].best_params_
        axes.flat[3].set_title(f"KNN: selected k = {p['model__n_neighbors']}, {p['model__weights']} weights")
    fig.suptitle("Hyperparameter tuning with 5-fold cross-validation (training set only)",
                 x=0.01, ha="left", fontweight="bold", fontsize=15)
    fig.tight_layout()
    return _save(fig, "07_cv_tuning_curves.png")


def pca_explained_variance(pcr_search, X_train) -> np.ndarray:
    """Explained-variance ratio of ALL components of the preprocessed training data."""
    from sklearn.decomposition import PCA
    prep = pcr_search.best_estimator_.named_steps["prep"]
    return PCA().fit(prep.transform(X_train)).explained_variance_ratio_


def plot_pca_variance(pcr_search, X_train):
    k = pcr_search.best_params_["pca__n_components"]
    evr = pca_explained_variance(pcr_search, X_train)
    cum = np.cumsum(evr)
    xs = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(xs, evr * 100, color="#9ec5f4", width=0.8, label="Individual component")
    ax.plot(xs, cum * 100, color=BLUE, lw=2.5, marker="o", ms=4, label="Cumulative")
    ax.axvline(k, color=ORANGE, ls="--", lw=2,
               label=f"Selected by CV: {k} components ({cum[k - 1] * 100:.1f}% of variance)")
    for t in (80, 90):
        n = int(np.searchsorted(cum, t / 100) + 1)
        ax.axhline(t, color=MUTED, ls=":", lw=1)
        ax.text(len(xs), t + 1, f"{t}% reached at {n} PCs", ha="right", fontsize=10, color=INK2)
    ax.set(title="PCA: cumulative explained variance of the standardised predictors",
           xlabel="Principal component", ylabel="Variance explained (%)", ylim=(0, 105))
    ax.legend(loc="lower right")
    return _save(fig, "08_pca_cumulative_variance.png")


def plot_model_comparison(results: pd.DataFrame, best: str, n_test: int = 36):
    r = results.sort_values("Test RMSE")
    y = np.arange(len(r))
    # Clip the axis so the OLS CV outlier does not flatten every other bar.
    cap = max(r["Test RMSE"].max(), (r["CV RMSE"] + r["CV RMSE SD"])[r["CV RMSE"] < 3].max()) * 1.25
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = [BLUE if m == best else "#9ec5f4" for m in r.index]
    ax.barh(y + 0.2, r["Test RMSE"], height=0.38, color=colors, label="Test RMSE")
    cv = r["CV RMSE"].clip(upper=cap)
    sd = r["CV RMSE SD"].where(r["CV RMSE"] < cap, 0)
    ax.barh(y - 0.2, cv, height=0.38, color="#d8d7d0", xerr=sd,
            error_kw={"ecolor": MUTED, "capsize": 3}, label="CV RMSE (±1 SD)")
    for i, m in enumerate(r.index):
        t, c, e = r.loc[m, "Test RMSE"], r.loc[m, "CV RMSE"], r.loc[m, "CV RMSE SD"]
        ax.text(t + 0.03, i + 0.2, f"{t:.3f}", va="center", fontsize=10, color=INK)
        if c >= cap:
            ax.text(cap - 0.03, i - 0.2, f"{c:.2f} (off scale, SD {e:.1f}) →", va="center",
                    ha="right", fontsize=10, color=INK)
        else:
            ax.text(c + e + 0.04, i - 0.2, f"{c:.3f}", va="center", fontsize=10, color=INK2)
    ax.set_yticks(y, [f"{m} (selected)" if m == best else m for m in r.index])
    ax.invert_yaxis()
    ax.set(title="Model comparison: prediction error (lower is better)",
           xlabel="RMSE (yield percentage points)", xlim=(0, cap))
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.text(0, -0.30, f"Selected model = lowest CV RMSE on training data. Test set: {n_test} batches.",
            transform=ax.transAxes, fontsize=10, color=INK2)
    return _save(fig, "09_model_comparison_rmse.png")


def plot_actual_vs_predicted(y_true, y_pred, model_name: str, metrics: dict):
    fig, ax = plt.subplots(figsize=(7, 7))
    lo = min(np.min(y_true), np.min(y_pred)) - 0.5
    hi = max(np.max(y_true), np.max(y_pred)) + 0.5
    ax.plot([lo, hi], [lo, hi], color=MUTED, ls="--", lw=1.5, label="Perfect prediction")
    ax.scatter(y_true, y_pred, s=60, color=BLUE, edgecolor="white", lw=1.5, label="Test batch")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual yield (%)", ylabel="Predicted yield (%)",
           title=f"{model_name}: actual vs predicted (test set)")
    ax.set_aspect("equal")
    ax.text(0.04, 0.96, f"RMSE = {metrics['RMSE']:.3f}\nMAE = {metrics['MAE']:.3f}\n"
            f"R² = {metrics['R²']:.3f}\nn = {len(y_true)}", transform=ax.transAxes,
            va="top", fontsize=11, color=INK, bbox={"fc": "white", "ec": GRID})
    ax.legend(loc="lower right")
    return _save(fig, "10_actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, model_name: str):
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(y_pred, resid, s=55, color=BLUE, edgecolor="white", lw=1.5)
    ax1.axhline(0, color=ORANGE, lw=2, ls="--")
    ax1.set(title="Residuals vs predicted", xlabel="Predicted yield (%)",
            ylabel="Residual (actual − predicted)")
    (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
    ax2.scatter(osm, osr, s=55, color=BLUE, edgecolor="white", lw=1.5)
    ax2.plot(osm, slope * np.asarray(osm) + intercept, color=ORANGE, lw=2)
    ax2.set(title="Normal Q–Q plot of residuals", xlabel="Theoretical quantiles",
            ylabel="Ordered residuals")
    fig.suptitle(f"{model_name}: residual diagnostics on the test set", x=0.01, ha="left",
                 fontweight="bold", fontsize=15)
    fig.tight_layout()
    return _save(fig, "11_residual_plot.png")


def plot_ridge_vs_lasso(coefs: pd.DataFrame, n: int = 20):
    top = coefs.reindex(coefs[["Ridge", "Lasso"]].abs().max(axis=1)
                        .sort_values(ascending=False).index[:n])
    top = top.iloc[::-1]
    labels = [c.replace("ManufacturingProcess", "MP").replace("BiologicalMaterial", "BM")
              for c in top.index]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(y + 0.2, top["Ridge"], height=0.38, color=BLUE, label="Ridge (L2)")
    ax.barh(y - 0.2, top["Lasso"], height=0.38, color=ORANGE, label="Lasso (L1)")
    for i, v in enumerate(top["Lasso"]):
        if v == 0:
            ax.text(0.005, i - 0.2, "0 (dropped)", va="center", fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, labels)
    n_zero = int((coefs["Lasso"] == 0).sum())
    ax.set(title=f"Ridge vs Lasso standardised coefficients (top {n})",
           xlabel="Change in yield per 1 SD increase in predictor")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    ax.text(0, -0.11, f"Lasso set {n_zero} of {len(coefs)} coefficients exactly to zero; "
            "Ridge kept all of them non-zero.", transform=ax.transAxes, fontsize=10, color=INK2)
    return _save(fig, "12_ridge_vs_lasso_coefficients.png")


def plot_ols_diagnostics(ols_search, X_train, y_train):
    """Training-set residual diagnostics for the full OLS baseline."""
    fitted = ols_search.predict(X_train)
    resid = y_train - fitted
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(fitted, resid, s=40, color=BLUE, edgecolor="white", lw=1)
    ax1.axhline(0, color=ORANGE, ls="--", lw=2)
    ax1.set(title="Residuals vs fitted (training data)", xlabel="Fitted yield (%)", ylabel="Residual")
    stats.probplot(resid, dist="norm", plot=ax2)
    ax2.get_lines()[0].set(color=BLUE, markersize=5)
    ax2.get_lines()[1].set(color=ORANGE, lw=2)
    ax2.set_title("Normal Q–Q plot (training residuals)")
    fig.suptitle("Multiple linear regression (all predictors): training-set diagnostics",
                 x=0.01, ha="left", fontweight="bold", fontsize=15)
    fig.tight_layout()
    return _save(fig, "13_ols_residual_diagnostics.png")


def plot_grouped_validation(results: pd.DataFrame, grouped: pd.DataFrame):
    models = list(results.index)
    y = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y - 0.2, results.loc[models, "Test RMSE"], height=0.38, color=BLUE,
            label="Random 80/20 split")
    ax.barh(y + 0.2, grouped.loc[models, "Grouped test RMSE"], height=0.38, color=ORANGE,
            label="Batch-grouped 80/20 split")
    for i, m in enumerate(models):
        ax.text(results.loc[m, "Test RMSE"] + 0.03, i - 0.2, f"{results.loc[m, 'Test RMSE']:.3f}",
                va="center", fontsize=10)
        ax.text(grouped.loc[m, "Grouped test RMSE"] + 0.03, i + 0.2,
                f"{grouped.loc[m, 'Grouped test RMSE']:.3f}", va="center", fontsize=10)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set(title="Sensitivity check: does sharing raw-material batches inflate accuracy?",
           xlabel="Test RMSE (yield percentage points)")
    ax.set_xlim(0, max(results["Test RMSE"].max(), grouped["Grouped test RMSE"].max()) * 1.18)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    return _save(fig, "14_grouped_validation.png")


def plot_repeated_splits(repeated: pd.DataFrame):
    order = repeated.groupby("Model")["Test RMSE"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=repeated, y="Model", x="Test RMSE", order=order, color="#cde2fb",
                width=0.55, showfliers=False, ax=ax)
    sns.stripplot(data=repeated, y="Model", x="Test RMSE", order=order, color=BLUE, size=5,
                  jitter=0.15, ax=ax)
    n = repeated["seed"].nunique()
    ax.set(title=f"Test RMSE across {n} different random train/test splits",
           xlabel="Test RMSE (yield percentage points)", ylabel="")
    ax.grid(axis="y", visible=False)
    return _save(fig, "15_repeated_split_rmse.png")
