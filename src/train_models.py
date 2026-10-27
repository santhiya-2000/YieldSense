"""Model definitions and hyperparameter tuning.

Five regression approaches, each wrapped in a leakage-safe Pipeline and tuned
with 5-fold cross-validation on the training set only:

1. Multiple linear regression (OLS)     - no tuning (baseline)
2. Ridge regression (L2)                - alpha
3. Lasso regression (L1)                - alpha
4. Principal component regression (PCR) - number of components
5. K-nearest-neighbours regression      - k and distance weighting

Run the whole project end to end with:  python -m src.train_models
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     KFold, cross_val_score)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from src.preprocessing import RANDOM_STATE, build_preprocessor

N_FOLDS = 5
SCORING = "neg_root_mean_squared_error"
MODEL_ORDER = ["Linear Regression", "Ridge", "Lasso", "PCR", "KNN Regression"]


def make_cv(random_state: int = RANDOM_STATE) -> KFold:
    """Shuffled 5-fold CV with a fixed seed so results are reproducible."""
    return KFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)


def model_specs() -> dict[str, tuple[Pipeline, dict]]:
    """Return {name: (pipeline, parameter grid)} for the five models."""
    return {
        "Linear Regression": (
            Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())]),
            {},
        ),
        "Ridge": (
            Pipeline([("prep", build_preprocessor()), ("model", Ridge())]),
            {"model__alpha": np.logspace(-2, 4, 61)},
        ),
        "Lasso": (
            Pipeline([("prep", build_preprocessor()),
                      ("model", Lasso(max_iter=100_000, random_state=RANDOM_STATE))]),
            {"model__alpha": np.logspace(-3, 0, 61)},
        ),
        "PCR": (
            Pipeline([("prep", build_preprocessor()), ("pca", PCA()),
                      ("model", LinearRegression())]),
            {"pca__n_components": list(range(1, 41))},
        ),
        "KNN Regression": (
            Pipeline([("prep", build_preprocessor()), ("model", KNeighborsRegressor())]),
            {"model__n_neighbors": list(range(1, 31)),
             "model__weights": ["uniform", "distance"]},
        ),
    }


def tune_model(pipeline: Pipeline, grid: dict, X, y, cv=None, groups=None) -> GridSearchCV:
    """Grid-search one pipeline with CV RMSE; refit the best on all training data.

    With an empty grid this still runs CV, so the baseline gets a CV RMSE too.
    """
    cv = cv if cv is not None else make_cv()
    search = GridSearchCV(pipeline, grid or {}, scoring=SCORING, cv=cv, n_jobs=-1,
                          refit=True, return_train_score=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        search.fit(X, y, groups=groups)
    return search


def tune_all(X, y, cv=None, groups=None) -> dict[str, GridSearchCV]:
    """Tune every model on the training data."""
    return {name: tune_model(pipe, grid, X, y, cv=cv, groups=groups)
            for name, (pipe, grid) in model_specs().items()}


def best_cv_summary(searches: dict[str, GridSearchCV]) -> pd.DataFrame:
    """CV mean/sd RMSE and train RMSE at each model's selected hyperparameters."""
    rows = []
    for name, s in searches.items():
        i = s.best_index_
        cvr = s.cv_results_
        params = {k.split("__")[-1]: v for k, v in s.best_params_.items()}
        rows.append({
            "Model": name,
            "CV RMSE": -cvr["mean_test_score"][i],
            "CV RMSE SD": cvr["std_test_score"][i],
            "CV train RMSE": -cvr["mean_train_score"][i],
            "Selected hyperparameters": ", ".join(f"{k}={_fmt(v)}" for k, v in params.items()) or "none",
        })
    return pd.DataFrame(rows).set_index("Model")


def select_best_model(searches: dict[str, GridSearchCV]) -> str:
    """Pick the model with the lowest CV RMSE (the test set plays no part)."""
    return min(searches, key=lambda n: -searches[n].best_score_)


def _fmt(v):
    return f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)


# ------------------------------------------------------- sensitivity analyses

def grouped_cv_comparison(searches: dict[str, GridSearchCV], X, y, groups) -> pd.DataFrame:
    """Re-score each selected model with random KFold vs GroupKFold (by batch).

    GroupKFold keeps every biological-material profile entirely inside one
    fold, so the model is never validated on a batch it has partly seen.
    """
    rows = []
    for name, s in searches.items():
        est = s.best_estimator_
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            rand = -cross_val_score(est, X, y, cv=make_cv(), scoring=SCORING, n_jobs=-1)
            grp = -cross_val_score(est, X, y, cv=GroupKFold(n_splits=N_FOLDS),
                                   groups=groups, scoring=SCORING, n_jobs=-1)
        rows.append({"Model": name, "Random 5-fold CV RMSE": rand.mean(),
                     "Grouped 5-fold CV RMSE": grp.mean(),
                     "Increase (%)": 100 * (grp.mean() / rand.mean() - 1)})
    return pd.DataFrame(rows).set_index("Model")


def grouped_holdout_evaluation(X, y, groups, test_size: float = 0.20) -> pd.DataFrame:
    """Full re-run with a batch-grouped 80/20 split and GroupKFold tuning."""
    from src.evaluate_models import regression_metrics

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    tr, te = next(gss.split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    searches = tune_all(Xtr, ytr, cv=GroupKFold(n_splits=N_FOLDS), groups=groups.iloc[tr])
    rows = []
    for name, s in searches.items():
        m = regression_metrics(yte, s.predict(Xte))
        rows.append({"Model": name, "Grouped CV RMSE": -s.best_score_,
                     "Grouped test RMSE": m["RMSE"], "Grouped test R²": m["R²"]})
    out = pd.DataFrame(rows).set_index("Model")
    out.attrs["n_train"], out.attrs["n_test"] = len(tr), len(te)
    return out


def repeated_split_evaluation(X, y, n_repeats: int = 20) -> pd.DataFrame:
    """Repeat the whole tune-then-test procedure over different random splits.

    With only ~35 test rows, one split's test RMSE is noisy. This shows how
    much the ranking of models depends on which rows land in the test set.
    """
    from sklearn.model_selection import train_test_split

    from src.evaluate_models import regression_metrics

    rows = []
    for seed in range(n_repeats):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        for name, s in tune_all(Xtr, ytr, cv=make_cv(seed)).items():
            rows.append({"seed": seed, "Model": name,
                         "Test RMSE": regression_metrics(yte, s.predict(Xte))["RMSE"]})
    return pd.DataFrame(rows)


def main() -> None:
    """Run the complete analysis and write every figure and table."""
    from src.evaluate_models import run_full_analysis

    run_full_analysis(verbose=True)


if __name__ == "__main__":
    main()
