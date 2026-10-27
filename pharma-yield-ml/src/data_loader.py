"""Loading and validating the ChemicalManufacturingProcess dataset.

Source: Kuhn, M. & Johnson, K. (2013). Applied Predictive Modeling. Springer.
Distributed in the R package ``AppliedPredictiveModeling`` as
``ChemicalManufacturingProcess``. A CSV export is mirrored at DATA_URL.

The loader never simulates data. It reads ``data/ChemicalManufacturingProcess.csv``
if present, otherwise downloads it once from DATA_URL. To use a manually
obtained copy, place the CSV at that path or pass ``path=...``.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "ChemicalManufacturingProcess.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

DATA_URL = (
    "https://raw.githubusercontent.com/rwepa/DataDemo/refs/heads/master/"
    "ChemicalManufacturingProcess.csv"
)
TARGET = "Yield"
EXPECTED_N_BIO = 12
EXPECTED_N_PROCESS = 45


def download_data(url: str = DATA_URL, dest: Path = DATA_PATH) -> Path:
    """Download the CSV to ``dest`` and return its path."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:  # network errors, HTTP errors
        raise RuntimeError(
            f"Could not download the dataset from {url} ({exc}).\n"
            f"Download it manually (or export it in R with\n"
            f"  library(AppliedPredictiveModeling); data(ChemicalManufacturingProcess);\n"
            f"  write.csv(ChemicalManufacturingProcess, 'ChemicalManufacturingProcess.csv', row.names=FALSE))\n"
            f"and save it as {dest}."
        ) from exc
    return dest


def load_data(path: str | Path | None = None, download: bool = True) -> pd.DataFrame:
    """Load the dataset, downloading it if it is not on disk yet."""
    path = Path(path) if path is not None else DATA_PATH
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found and download=False.")
        download_data(dest=path)
    df = pd.read_csv(path)
    validate_data(df)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Check the structure we rely on; raise ValueError if it does not hold."""
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' missing.")
    if df[TARGET].isna().any():
        raise ValueError("Target column contains missing values.")
    bio, proc = feature_groups(df)
    if len(bio) != EXPECTED_N_BIO or len(proc) != EXPECTED_N_PROCESS:
        raise ValueError(
            f"Expected {EXPECTED_N_BIO} biological and {EXPECTED_N_PROCESS} process "
            f"columns, found {len(bio)} and {len(proc)}."
        )
    non_numeric = df.columns[~df.dtypes.apply(pd.api.types.is_numeric_dtype)]
    if len(non_numeric):
        raise ValueError(f"Non-numeric columns: {list(non_numeric)}")


def feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (biological-material columns, manufacturing-process columns)."""
    bio = [c for c in df.columns if c.startswith("BiologicalMaterial")]
    proc = [c for c in df.columns if c.startswith("ManufacturingProcess")]
    return bio, proc


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the predictors X from the target y."""
    return df.drop(columns=TARGET), df[TARGET]


def batch_groups(df: pd.DataFrame) -> pd.Series:
    """Label rows that share an identical biological-material profile.

    The 12 biological measurements describe the raw material that went into a
    batch. Rows with exactly the same 12 values almost certainly used the same
    raw-material lot, so they are not independent observations. Each distinct
    profile gets an integer group id.
    """
    bio, _ = feature_groups(df)
    key = df[bio].round(6).astype(str).agg("|".join, axis=1)
    return pd.Series(pd.factorize(key)[0], index=df.index, name="bio_group")


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary of shape, feature groups and missingness."""
    bio, proc = feature_groups(df)
    groups = batch_groups(df)
    sizes = groups.value_counts()
    return pd.DataFrame(
        {
            "observations": [len(df)],
            "predictors": [df.shape[1] - 1],
            "biological_material": [len(bio)],
            "manufacturing_process": [len(proc)],
            "missing_cells": [int(df.isna().sum().sum())],
            "columns_with_missing": [int(df.isna().any().sum())],
            "rows_with_missing": [int(df.isna().any(axis=1).sum())],
            "distinct_bio_profiles": [groups.nunique()],
            "rows_in_shared_profiles": [int(sizes[sizes > 1].sum())],
        }
    )
