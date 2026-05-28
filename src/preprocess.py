"""Preprocessing and validation utilities for the Aero Toolkit dataset.

This module should be used before training. It keeps the project schema in one
place so the notebook, training script, inference code, and Streamlit app all
use the same expected inputs.

Important design rule:
    Validation and cleaning are separated.

The model training workflow should validate that the CSV is already clean. The
helper `standardize_raw_dataset` can be used intentionally when a raw file needs
minor formatting standardization, but the training code should not silently
rewrite scientific labels or targets.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from src.utils import dataframe_summary
except ModuleNotFoundError:
    from utils import dataframe_summary


TARGET_COLUMN = "separation_x_over_c"

FEATURE_COLUMNS = [
    "airfoil_family",
    "tubercle_amplitude",
    "tubercle_wavelength",
    "tubercle_shape",
    "root_chord",
    "tip_chord",
    "sweep_angle",
    "angle_of_attack",
    "airspeed",
]

CATEGORICAL_FEATURES = ["airfoil_family", "tubercle_shape"]
NUMERIC_FEATURES = [col for col in FEATURE_COLUMNS if col not in CATEGORICAL_FEATURES]
OPTIONAL_METADATA_COLUMNS = ["data_source", "simulation_id", "notes"]

VALID_AIRFOIL_FAMILIES = {"symmetric", "cambered", "biomimetic"}
VALID_TUBERCLE_SHAPES = {"none", "whale", "biomimetic_v1"}

COLUMN_ALIASES = {
    "separation_x/c": TARGET_COLUMN,
    "separation_x_over_c": TARGET_COLUMN,
    "separation x over c": TARGET_COLUMN,
    "separation x/c": TARGET_COLUMN,
    "flow_separation_x_over_c": TARGET_COLUMN,
    "flow separation x over c": TARGET_COLUMN,
    "aoa": "angle_of_attack",
    "angle of attack": "angle_of_attack",
    "angle-of-attack": "angle_of_attack",
    "velocity": "airspeed",
    "air_speed": "airspeed",
    "air speed": "airspeed",
    "root chord": "root_chord",
    "tip chord": "tip_chord",
    "sweep angle": "sweep_angle",
    "tubercle amplitude": "tubercle_amplitude",
    "tubercle wavelength": "tubercle_wavelength",
    "tubercle shape": "tubercle_shape",
    "airfoil family": "airfoil_family",
}


def normalize_column_name(column_name: str) -> str:
    """Normalize a single column name into snake_case."""
    original = str(column_name).strip()
    alias_key = original.lower()
    if alias_key in COLUMN_ALIASES:
        return COLUMN_ALIASES[alias_key]

    normalized = original.lower()
    normalized = normalized.replace("/", "_over_")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return COLUMN_ALIASES.get(normalized, normalized)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with normalized column names."""
    output = df.copy()
    output.columns = [normalize_column_name(col) for col in output.columns]
    return output


def load_dataset(csv_path: Path | str, normalize_columns: bool = True) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    if normalize_columns:
        df = normalize_column_names(df)
    return df


def standardize_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize formatting without inventing or guessing scientific values.

    This function intentionally performs only safe formatting operations:
    - column-name normalization
    - string trimming and lowercasing for categorical fields
    - numeric conversion for numeric fields and target

    It does not guess missing separation values. It does not relabel airfoil
    families based on tubercle values. Those decisions should be handled in the
    data-generation log or by explicit human review.
    """
    output = normalize_column_names(df)

    for col in CATEGORICAL_FEATURES:
        if col in output.columns:
            output[col] = output[col].astype(str).str.strip().str.lower()

    for col in NUMERIC_FEATURES + [TARGET_COLUMN]:
        if col in output.columns:
            output[col] = pd.to_numeric(output[col], errors="coerce")

    return output


def required_columns() -> List[str]:
    """Return all required columns, including the target."""
    return FEATURE_COLUMNS + [TARGET_COLUMN]


def find_missing_required_columns(df: pd.DataFrame) -> List[str]:
    """Return required columns that are missing from the dataset."""
    return [col for col in required_columns() if col not in df.columns]


def detect_constant_columns(df: pd.DataFrame, columns: List[str] | None = None) -> List[str]:
    """Return columns with one or fewer unique non-missing values."""
    if columns is None:
        columns = list(df.columns)

    constant_columns = []
    for col in columns:
        if col in df.columns and df[col].nunique(dropna=True) <= 1:
            constant_columns.append(col)
    return constant_columns


def validate_dataset(df: pd.DataFrame, strict: bool = True) -> Dict[str, Any]:
    """Validate the dataset schema and values.

    Parameters
    ----------
    df:
        Dataset after safe standardization.
    strict:
        If True, raise ValueError when validation fails. If False, return the
        report without raising.
    """
    errors: List[str] = []
    warnings: List[str] = []

    missing_columns = find_missing_required_columns(df)
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))

    if errors:
        report = {"errors": errors, "warnings": warnings, "summary": dataframe_summary(df)}
        if strict:
            raise ValueError("Dataset validation failed: " + " | ".join(errors))
        return report

    missing_counts = df[required_columns()].isna().sum()
    columns_with_missing = missing_counts[missing_counts > 0].to_dict()
    if columns_with_missing:
        errors.append(f"Missing values found in required columns: {columns_with_missing}")

    invalid_airfoil = sorted(set(df["airfoil_family"].dropna()) - VALID_AIRFOIL_FAMILIES)
    if invalid_airfoil:
        errors.append(
            "Invalid airfoil_family values: "
            + ", ".join(map(str, invalid_airfoil))
            + f". Expected: {sorted(VALID_AIRFOIL_FAMILIES)}"
        )

    invalid_shapes = sorted(set(df["tubercle_shape"].dropna()) - VALID_TUBERCLE_SHAPES)
    if invalid_shapes:
        errors.append(
            "Invalid tubercle_shape values: "
            + ", ".join(map(str, invalid_shapes))
            + f". Expected: {sorted(VALID_TUBERCLE_SHAPES)}"
        )

    if not df[TARGET_COLUMN].between(0, 1, inclusive="both").all():
        errors.append(f"{TARGET_COLUMN} must be between 0 and 1 for every row.")

    for col in ["tubercle_amplitude", "tubercle_wavelength"]:
        if (df[col] < 0).any():
            errors.append(f"{col} cannot contain negative values.")

    for col in ["root_chord", "tip_chord", "airspeed"]:
        if (df[col] <= 0).any():
            errors.append(f"{col} must be greater than 0 for every row.")

    non_biomimetic = df["airfoil_family"].isin(["symmetric", "cambered"])
    non_biomimetic_with_tubercles = df.loc[
        non_biomimetic
        & (
            (df["tubercle_amplitude"] != 0)
            | (df["tubercle_wavelength"] != 0)
            | (df["tubercle_shape"] != "none")
        )
    ]
    if len(non_biomimetic_with_tubercles) > 0:
        warnings.append(
            "Some symmetric or cambered rows have nonzero tubercle fields. "
            "Review these rows before final training."
        )

    biomimetic = df["airfoil_family"] == "biomimetic"
    biomimetic_without_tubercles = df.loc[
        biomimetic
        & (
            (df["tubercle_amplitude"] <= 0)
            | (df["tubercle_wavelength"] <= 0)
            | (df["tubercle_shape"] == "none")
        )
    ]
    if len(biomimetic_without_tubercles) > 0:
        warnings.append(
            "Some biomimetic rows have zero or missing tubercle information. "
            "Review these rows before final training."
        )

    constant_features = detect_constant_columns(df, FEATURE_COLUMNS)
    if constant_features:
        warnings.append(
            "Constant feature columns detected and should be dropped during training: "
            + ", ".join(constant_features)
        )

    duplicate_count = int(df.duplicated(subset=FEATURE_COLUMNS).sum())
    if duplicate_count > 0:
        warnings.append(
            f"Found {duplicate_count} duplicated feature rows. This may be okay if simulations were repeated, "
            "but it should be intentional."
        )

    report = {
        "errors": errors,
        "warnings": warnings,
        "summary": dataframe_summary(df),
        "constant_features": constant_features,
        "duplicate_feature_rows": duplicate_count,
    }

    if strict and errors:
        raise ValueError("Dataset validation failed: " + " | ".join(errors))

    return report


def prepare_modeling_table(
    df: pd.DataFrame,
    drop_constant_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """Return X, y, feature columns used, and a preprocessing report."""
    standardized = standardize_raw_dataset(df)
    report = validate_dataset(standardized, strict=True)

    feature_columns_used = FEATURE_COLUMNS.copy()
    if drop_constant_features:
        constant_features = detect_constant_columns(standardized, FEATURE_COLUMNS)
        feature_columns_used = [col for col in feature_columns_used if col not in constant_features]
        report["dropped_constant_features"] = constant_features
    else:
        report["dropped_constant_features"] = []

    X = standardized[feature_columns_used].copy()
    y = standardized[TARGET_COLUMN].copy()
    return X, y, feature_columns_used, report


def get_feature_groups(feature_columns: List[str]) -> Tuple[List[str], List[str]]:
    """Return numeric and categorical features that are actually used."""
    categorical = [col for col in feature_columns if col in CATEGORICAL_FEATURES]
    numeric = [col for col in feature_columns if col not in categorical]
    return numeric, categorical


def build_preprocessor(feature_columns: List[str]) -> ColumnTransformer:
    """Build the preprocessing transformer used inside sklearn pipelines.

    Numeric features are imputed and scaled. Categorical features are imputed and
    one-hot encoded. `handle_unknown='ignore'` prevents the app from crashing if
    a future valid category is added before the model is retrained.
    """
    numeric_features, categorical_features = get_feature_groups(feature_columns)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def load_and_prepare_dataset(
    csv_path: Path | str,
    drop_constant_features: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """Load, safely standardize, validate, and split a dataset into X and y."""
    df = load_dataset(csv_path, normalize_columns=True)
    return prepare_modeling_table(df, drop_constant_features=drop_constant_features)


def make_single_input_dataframe(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Create a one-row DataFrame from app-style inputs using the full V1 schema."""
    missing = [col for col in FEATURE_COLUMNS if col not in input_dict]
    if missing:
        raise ValueError("Missing input fields: " + ", ".join(missing))

    row = {col: input_dict[col] for col in FEATURE_COLUMNS}
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return standardize_raw_dataset(df)
