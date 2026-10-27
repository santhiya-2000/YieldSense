"""Preprocessing: train/test split, leakage-safe transformers, EDA helpers.

Every step that learns from data (near-zero-variance filter, median imputer,
scaler, PCA) lives inside a scikit-learn Pipeline. GridSearchCV therefore
refits them on the training folds only, and the test set is never seen
until the final evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42
TEST_SIZE = 0.20


def make_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE,
                          random_state: int = RANDOM_STATE):
    """Reproducible 80/20 random split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


class NearZeroVarianceFilter(BaseEstimator, TransformerMixin):
    """Drop predictors that are (almost) constant.

    Uses the rule from Kuhn & Johnson (2013): a column is removed when the most
    common value is more than ``freq_cut`` times as frequent as the second most
    common value AND the share of distinct values is below ``unique_cut``
    percent. Such columns carry almost no information and make some CV folds
    contain a constant column. Fitted on training data only.
    """

    def __init__(self, freq_cut: float = 95 / 5, unique_cut: float = 10.0):
        self.freq_cut = freq_cut
        self.unique_cut = unique_cut

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        keep = []
        for col in X.columns:
            counts = X[col].dropna().value_counts()
            if len(counts) <= 1:
                keep.append(False)
                continue
            freq_ratio = counts.iloc[0] / counts.iloc[1]
            pct_unique = 100 * len(counts) / X[col].notna().sum()
            keep.append(not (freq_ratio > self.freq_cut and pct_unique < self.unique_cut))
        self.support_ = np.array(keep)
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        if isinstance(X, pd.DataFrame):
            return X.loc[:, self.support_]
        return np.asarray(X)[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        names = self.feature_names_in_ if input_features is None else np.asarray(input_features)
        return names[self.support_]

    @property
    def removed_features_(self) -> list[str]:
        return list(self.feature_names_in_[~self.support_])


def build_preprocessor(scale: bool = True) -> Pipeline:
    """Near-zero-variance filter -> median imputation -> (optional) standardisation."""
    steps = [
        ("nzv", NearZeroVarianceFilter()),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


# --------------------------------------------------------------------------- EDA

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Missing-value count and percentage for columns that have any."""
    counts = df.isna().sum()
    out = pd.DataFrame({"missing": counts, "percent": 100 * counts / len(df)})
    return out[out.missing > 0].sort_values("missing", ascending=False)


def correlated_pairs(X: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Predictor pairs with |Pearson r| >= threshold (pairwise-complete)."""
    corr = X.corr()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().rename("r").reset_index()
    pairs.columns = ["feature_1", "feature_2", "r"]
    pairs = pairs[pairs.r.abs() >= threshold]
    return pairs.reindex(pairs.r.abs().sort_values(ascending=False).index).reset_index(drop=True)


def target_correlations(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Pearson correlation of each predictor with the target, sorted by |r|."""
    r = X.corrwith(y)
    return r.reindex(r.abs().sort_values(ascending=False).index)


def select_uncorrelated_top_features(X: pd.DataFrame, y: pd.Series, k: int = 8,
                                     max_inter_corr: float = 0.70) -> list[str]:
    """Greedy pick of up to k features most correlated with y.

    A candidate is skipped if |r| with an already chosen feature exceeds
    ``max_inter_corr``. Used to build a small, interpretable OLS model whose
    coefficients are not wrecked by multicollinearity. Call on TRAINING data only.
    """
    ranked = target_correlations(X, y).index
    corr = X.corr().abs()
    chosen: list[str] = []
    for feat in ranked:
        if all(corr.loc[feat, c] <= max_inter_corr for c in chosen):
            chosen.append(feat)
        if len(chosen) == k:
            break
    return chosen
