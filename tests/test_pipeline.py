"""Basic tests for data loading, preprocessing, leakage safety and models.

Run with:  python -m pytest -q
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold

from src import data_loader as dl
from src import evaluate_models as ev
from src import preprocessing as pp
from src import train_models as tm


@pytest.fixture(scope="module")
def df():
    return dl.load_data()


@pytest.fixture(scope="module")
def split(df):
    X, y = dl.split_xy(df)
    return pp.make_train_test_split(X, y)


# ------------------------------------------------------------------ data

def test_dataset_shape_and_columns(df):
    assert df.shape == (176, 58)
    bio, proc = dl.feature_groups(df)
    assert len(bio) == 12 and len(proc) == 45
    assert df[dl.TARGET].notna().all()
    assert df.drop(columns=dl.TARGET).isna().sum().sum() == 106


def test_validate_rejects_missing_target(df):
    with pytest.raises(ValueError):
        dl.validate_data(df.drop(columns=dl.TARGET))


def test_batch_groups(df):
    g = dl.batch_groups(df)
    assert g.nunique() == 108
    assert g.value_counts().max() == 3


# ------------------------------------------------------------ preprocessing

def test_split_is_reproducible_80_20(split, df):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 140 and len(X_test) == 36
    assert set(X_train.index).isdisjoint(X_test.index)
    X, y = dl.split_xy(df)
    again = pp.make_train_test_split(X, y)
    assert list(again[0].index) == list(X_train.index)


def test_near_zero_variance_filter():
    X = pd.DataFrame({"const": [1.0] * 100, "rare": [0.0] * 98 + [1.0] * 2,
                      "ok": np.arange(100.0)})
    f = pp.NearZeroVarianceFilter().fit(X)
    assert f.removed_features_ == ["const", "rare"]
    assert list(f.transform(X).columns) == ["ok"]


def test_preprocessor_output_has_no_nans_and_is_scaled(split):
    X_train, X_test, *_ = split
    prep = pp.build_preprocessor().fit(X_train)
    Z = prep.transform(X_train)
    assert not np.isnan(Z).any()
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-8)
    assert not np.isnan(prep.transform(X_test)).any()


def test_imputer_uses_training_medians_only(split):
    """Leakage check: imputation statistics must come from training rows."""
    X_train, X_test, *_ = split
    prep = pp.build_preprocessor(scale=False).fit(X_train)
    kept = prep.named_steps["nzv"].get_feature_names_out()
    expected = X_train[kept].median().values
    assert np.allclose(prep.named_steps["impute"].statistics_, expected)


def test_ols_subset_has_low_collinearity(split):
    X_train, _, y_train, _ = split
    feats = pp.select_uncorrelated_top_features(X_train, y_train, k=8, max_inter_corr=0.7)
    assert len(feats) == 8
    corr = X_train[feats].corr().abs().values
    assert corr[np.triu_indices(8, k=1)].max() <= 0.7


# ----------------------------------------------------------------- models

def test_model_specs_are_pipelines_with_preprocessing():
    specs = tm.model_specs()
    assert list(specs) == tm.MODEL_ORDER
    for pipe, _ in specs.values():
        assert pipe.steps[0][0] == "prep"


def test_tuning_and_metrics(split):
    X_train, X_test, y_train, y_test = split
    pipe, grid = tm.model_specs()["Lasso"]
    search = tm.tune_model(pipe, {"model__alpha": [0.05, 0.1]}, X_train, y_train,
                           cv=KFold(3, shuffle=True, random_state=0))
    m = ev.regression_metrics(y_test, search.predict(X_test))
    assert set(m) == {"MSE", "RMSE", "MAE", "R²"}
    assert np.isclose(m["RMSE"] ** 2, m["MSE"])
    # A sensible model beats predicting the training mean.
    baseline = ev.regression_metrics(y_test, np.full(len(y_test), y_train.mean()))
    assert m["RMSE"] < baseline["RMSE"]


def test_pcr_coefficients_match_predictions(split):
    """Back-transformed PCR coefficients must reproduce the pipeline's predictions."""
    X_train, X_test, y_train, _ = split
    pipe, _ = tm.model_specs()["PCR"]
    search = tm.tune_model(pipe, {"pca__n_components": [5]}, X_train, y_train,
                           cv=KFold(3, shuffle=True, random_state=0))
    est = search.best_estimator_
    Z = est.named_steps["prep"].transform(X_test)
    manual = Z @ ev.pcr_coefficients(search).values + (
        est.named_steps["model"].intercept_
        - est.named_steps["pca"].mean_ @ ev.pcr_coefficients(search).values)
    assert np.allclose(manual, search.predict(X_test))


def test_metrics_helper_known_values():
    m = ev.regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    assert np.isclose(m["MSE"], 4 / 3)
    assert np.isclose(m["MAE"], 2 / 3)
