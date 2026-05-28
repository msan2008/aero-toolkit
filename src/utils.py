"""Utility helpers for the Aero Toolkit project.

This file contains reusable project helpers for paths, metrics, saving model
artifacts, saving metadata, and exporting prediction tables.

Expected repo location:
    aero-toolkit/
        src/utils.py
        src/preprocess.py
        src/train_model.py
        data/processed/openfoam_phase1_cleaned.csv
        models/notebook2_gradient_boosting.joblib
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_project_root() -> Path:
    """Return the repository root.

    This assumes this file lives inside the src/ folder. If it is moved, the
    fallback returns the current working directory.
    """
    try:
        return Path(__file__).resolve().parents[1]
    except NameError:
        return Path.cwd().resolve()


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def ensure_directory(path: Path | str) -> Path:
    """Create a directory if it does not already exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp_utc() -> str:
    """Return a compact UTC timestamp for metadata and versioned files."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def print_section(title: str) -> None:
    """Print a clean console section header."""
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Compute root mean squared error in a version-stable way."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """Return standard regression metrics for model comparison."""
    y_true_array = np.asarray(list(y_true), dtype=float)
    y_pred_array = np.asarray(list(y_pred), dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": rmse(y_true_array, y_pred_array),
        "r2": float(r2_score(y_true_array, y_pred_array)),
    }


def save_json(data: Dict[str, Any], path: Path | str) -> Path:
    """Save a dictionary as a formatted JSON file."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    return output_path


def load_json(path: Path | str) -> Dict[str, Any]:
    """Load a JSON file as a dictionary."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_model(model: Any, path: Path | str) -> Path:
    """Save a fitted model or sklearn pipeline with joblib."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    joblib.dump(model, output_path)
    return output_path


def load_model(path: Path | str) -> Any:
    """Load a joblib model artifact."""
    return joblib.load(Path(path))


def save_model_and_metadata(
    model: Any,
    model_path: Path | str,
    metadata: Dict[str, Any],
    metadata_path: Optional[Path | str] = None,
) -> Dict[str, Path]:
    """Save a model artifact and its metadata JSON.

    Returns a dictionary with the saved paths so training scripts can print or
    reuse them.
    """
    saved_model_path = save_model(model, model_path)

    if metadata_path is None:
        metadata_path = saved_model_path.with_suffix(".metadata.json")

    metadata = dict(metadata)
    try:
        metadata["model_artifact"] = str(saved_model_path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        metadata["model_artifact"] = str(saved_model_path.resolve())
    metadata["saved_at_utc"] = timestamp_utc()
    metadata["python_version"] = platform.python_version()

    saved_metadata_path = save_json(metadata, metadata_path)
    return {"model": saved_model_path, "metadata": saved_metadata_path}


def find_first_existing(candidates: Iterable[Path | str]) -> Path:
    """Return the first path that exists from a list of candidate paths."""
    checked = []
    for candidate in candidates:
        path = Path(candidate)
        checked.append(str(path))
        if path.exists():
            return path

    raise FileNotFoundError(
        "None of the candidate files were found. Checked:\n" + "\n".join(checked)
    )


def export_holdout_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: Iterable[float],
    output_path: Path | str,
) -> Path:
    """Export a CSV containing holdout inputs, true values, predictions, and errors."""
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    predictions = X_test.copy().reset_index(drop=True)
    predictions["actual_separation_x_over_c"] = pd.Series(y_test).reset_index(drop=True)
    predictions["predicted_separation_x_over_c"] = np.asarray(list(y_pred), dtype=float)
    predictions["prediction_error"] = (
        predictions["predicted_separation_x_over_c"]
        - predictions["actual_separation_x_over_c"]
    )
    predictions["absolute_error"] = predictions["prediction_error"].abs()
    predictions.to_csv(output_path, index=False)
    return output_path


def dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a lightweight, JSON-safe summary of a DataFrame."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values_by_column": {col: int(df[col].isna().sum()) for col in df.columns},
    }
