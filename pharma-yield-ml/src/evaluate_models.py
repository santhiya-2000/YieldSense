"""Metrics, comparison tables, coefficient analysis and the end-to-end run."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src import data_loader as dl
from src import preprocessing as pp
from src import train_models as tm
from src import visualizations as viz


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": float(np.sqrt(mse)),
            "MAE": mean_absolute_error(y_true, y_pred), "R²": r2_score(y_true, y_pred)}


def comparison_table(searches, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """The main results table: CV RMSE plus test metrics for each tuned model."""
    cv = tm.best_cv_summary(searches)
    rows = []
    for name, s in searches.items():
        test = regression_metrics(y_test, s.predict(X_test))
        train = regression_metrics(y_train, s.predict(X_train))
        rows.append({"Model": name,
                     "CV RMSE": cv.loc[name, "CV RMSE"],
                     "CV RMSE SD": cv.loc[name, "CV RMSE SD"],
                     "Train RMSE": train["RMSE"],
                     "Test MSE": test["MSE"], "Test RMSE": test["RMSE"],
                     "Test MAE": test["MAE"], "Test R²": test["R²"],
                     "Selected hyperparameters": cv.loc[name, "Selected hyperparameters"]})
    return pd.DataFrame(rows).set_index("Model")


def linear_coefficients(search) -> pd.Series:
    """Standardised coefficients of a fitted linear pipeline, by feature name."""
    est = search.best_estimator_
    names = est.named_steps["prep"].get_feature_names_out()
    return pd.Series(est.named_steps["model"].coef_, index=names)


def pcr_coefficients(search) -> pd.Series:
    """Map PCR's component coefficients back to the standardised features."""
    est = search.best_estimator_
    names = est.named_steps["prep"].get_feature_names_out()
    beta = est.named_steps["pca"].components_.T @ est.named_steps["model"].coef_
    return pd.Series(beta, index=names)


def coefficient_table(searches) -> pd.DataFrame:
    """Standardised coefficients of OLS, Ridge, Lasso and PCR side by side."""
    tab = pd.DataFrame({
        "OLS": linear_coefficients(searches["Linear Regression"]),
        "Ridge": linear_coefficients(searches["Ridge"]),
        "Lasso": linear_coefficients(searches["Lasso"]),
        "PCR": pcr_coefficients(searches["PCR"]),
    })
    return tab.reindex(tab["Ridge"].abs().sort_values(ascending=False).index)


def removed_features(searches) -> list[str]:
    """Columns dropped by the near-zero-variance filter in the final fit."""
    return searches["Ridge"].best_estimator_.named_steps["prep"].named_steps["nzv"].removed_features_


def ols_statsmodels(X_train: pd.DataFrame, y_train: pd.Series, features: list[str]):
    """Interpretable statsmodels OLS on a small, low-collinearity subset.

    Predictors are median-imputed (training medians) and standardised so the
    coefficients are comparable: each is the change in yield for a one-SD
    change in that predictor, holding the others fixed.
    """
    Xs = X_train[features]
    Xs = Xs.fillna(Xs.median())
    Xs = (Xs - Xs.mean()) / Xs.std(ddof=0)
    Xc = sm.add_constant(Xs)
    fit = sm.OLS(y_train, Xc).fit()
    vif = pd.Series([variance_inflation_factor(Xc.values, i + 1) for i in range(len(features))],
                    index=features, name="VIF")
    return fit, vif


def full_ols_vif(X_train: pd.DataFrame, searches) -> pd.Series:
    """VIF for every predictor in the full OLS model (shows multicollinearity)."""
    prep = searches["Linear Regression"].best_estimator_.named_steps["prep"]
    Z = prep.transform(X_train)
    Zc = sm.add_constant(Z)
    with np.errstate(divide="ignore"):  # perfectly collinear columns give VIF = inf
        vif = [variance_inflation_factor(Zc, i + 1) for i in range(Z.shape[1])]
    return pd.Series(vif, index=prep.get_feature_names_out(), name="VIF").sort_values(ascending=False)


def overfitting_table(results: pd.DataFrame) -> pd.DataFrame:
    """Train vs CV vs test RMSE; a large train/CV gap signals overfitting."""
    out = results[["Train RMSE", "CV RMSE", "Test RMSE"]].copy()
    # Undefined when the model reproduces its training data exactly (e.g. distance-weighted KNN).
    train = out["Train RMSE"].where(out["Train RMSE"] > 1e-6)
    out["CV / Train ratio"] = out["CV RMSE"] / train
    return out


# -------------------------------------------------------------- end-to-end run

def run_full_analysis(verbose: bool = True, n_repeats: int = 20) -> dict:
    """Load data, run EDA, tune and evaluate all models, save figures/tables."""
    log = print if verbose else (lambda *a, **k: None)
    dl.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    dl.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    viz.set_style()

    df = dl.load_data()
    X, y = dl.split_xy(df)
    groups = dl.batch_groups(df)
    overview = dl.dataset_overview(df)
    log("Dataset overview:\n", overview.T.to_string(header=False))

    X_train, X_test, y_train, y_test = pp.make_train_test_split(X, y)
    log(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # Batch overlap between the random train and test sets.
    shared = set(groups[X_train.index]) & set(groups[X_test.index])
    overlap = int(groups[X_test.index].isin(shared).sum())
    log(f"Test rows whose biological batch also appears in training: {overlap}/{len(X_test)}")

    # ---- EDA (descriptive plots use all rows; anything that informs modelling uses train only)
    missing = pp.missing_summary(df)
    missing.to_csv(dl.TABLES_DIR / "missing_values.csv")
    df.describe().T.to_csv(dl.TABLES_DIR / "summary_statistics.csv")
    target_corr = pp.target_correlations(X_train, y_train)
    target_corr.rename("r_with_yield").to_csv(dl.TABLES_DIR / "target_correlations_train.csv")
    pairs = pp.correlated_pairs(X_train, 0.90)
    pairs.to_csv(dl.TABLES_DIR / "highly_correlated_pairs_train.csv", index=False)
    log(f"Predictor pairs with |r| >= 0.90 (train): {len(pairs)}")

    viz.plot_yield_distribution(y)
    viz.plot_missing_values(missing)
    top_feats = list(target_corr.index[:12])
    viz.plot_predictor_histograms(X_train, top_feats[:9])
    viz.plot_correlation_heatmap(X_train, y_train, top_feats)
    viz.plot_predictor_boxplots(X_train, top_feats)
    viz.plot_pairwise_correlation_hist(X_train)

    # ---- Model tuning on training data only
    log("\nTuning models with 5-fold CV ...")
    searches = tm.tune_all(X_train, y_train)
    results = comparison_table(searches, X_train, y_train, X_test, y_test)
    best = tm.select_best_model(searches)
    results.round(6).to_csv(dl.TABLES_DIR / "model_comparison.csv")
    log("\n", results.round(3).to_string())
    log(f"\nSelected model (lowest CV RMSE): {best}")

    coefs = coefficient_table(searches)
    coefs.to_csv(dl.TABLES_DIR / "standardized_coefficients.csv")
    dropped = removed_features(searches)
    log(f"Near-zero-variance features removed: {dropped}")

    best_pred = searches[best].predict(X_test)
    viz.plot_cv_curves(searches)
    viz.plot_pca_variance(searches["PCR"], X_train)
    viz.plot_model_comparison(results, best, n_test=len(y_test))
    viz.plot_actual_vs_predicted(y_test, best_pred, best, regression_metrics(y_test, best_pred))
    viz.plot_residuals(y_test, best_pred, best)
    viz.plot_ridge_vs_lasso(coefs, n=20)

    # ---- Interpretable OLS on a justified subset
    subset = pp.select_uncorrelated_top_features(X_train, y_train, k=8, max_inter_corr=0.70)
    ols_fit, ols_vif = ols_statsmodels(X_train, y_train, subset)
    (dl.TABLES_DIR / "ols_subset_summary.txt").write_text(ols_fit.summary().as_text())
    full_vif = full_ols_vif(X_train, searches)
    full_vif.to_csv(dl.TABLES_DIR / "vif_full_ols.csv")
    viz.plot_ols_diagnostics(searches["Linear Regression"], X_train, y_train)

    # ---- Sensitivity analyses
    log("\nGrouped (batch-aware) validation ...")
    grouped_cv = tm.grouped_cv_comparison(searches, X_train, y_train, groups[X_train.index])
    grouped_cv.round(4).to_csv(dl.TABLES_DIR / "grouped_cv_comparison.csv")
    grouped_hold = tm.grouped_holdout_evaluation(X, y, groups)
    grouped_hold.round(4).to_csv(dl.TABLES_DIR / "grouped_holdout_evaluation.csv")
    viz.plot_grouped_validation(results, grouped_hold)
    log(grouped_cv.round(3).to_string())
    log(grouped_hold.round(3).to_string())

    log(f"\nRepeated-split evaluation ({n_repeats} random splits) ...")
    repeated = tm.repeated_split_evaluation(X, y, n_repeats=n_repeats)
    repeated.to_csv(dl.TABLES_DIR / "repeated_split_test_rmse.csv", index=False)
    rep_summary = repeated.groupby("Model")["Test RMSE"].agg(["mean", "std", "min", "max"]).reindex(tm.MODEL_ORDER)
    wins = (repeated.loc[repeated.groupby("seed")["Test RMSE"].idxmin(), "Model"]
            .value_counts().reindex(tm.MODEL_ORDER, fill_value=0))
    rep_summary["times best"] = wins
    rep_summary.round(4).to_csv(dl.TABLES_DIR / "repeated_split_summary.csv")
    viz.plot_repeated_splits(repeated)
    log(rep_summary.round(3).to_string())

    log(f"\nFigures written to {dl.FIGURES_DIR}")
    return {
        "df": df, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "groups": groups, "overview": overview, "test_batch_overlap": overlap,
        "missing": missing, "pairs": pairs, "target_corr": target_corr,
        "searches": searches, "results": results, "best": best, "coefs": coefs,
        "dropped": dropped, "ols_fit": ols_fit, "ols_vif": ols_vif, "full_vif": full_vif,
        "ols_subset": subset, "grouped_cv": grouped_cv, "grouped_holdout": grouped_hold,
        "repeated": repeated, "repeated_summary": rep_summary,
    }


if __name__ == "__main__":
    run_full_analysis()
