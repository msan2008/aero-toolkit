"""Inference helpers for the Aero Toolkit Streamlit app.

This file is meant to live at:
    src/inference.py

It loads the saved scikit-learn pipeline from the repo's models/ folder,
validates Streamlit inputs, runs predictions, and creates sensitivity sweeps
so the app can show how the model responds when one variable changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
# Expected repo structure:
# aero-toolkit/
#   app/app.py
#   src/inference.py
#   models/notebook2_gradient_boosting.joblib
#   requirements.txt
#
# __file__ points to src/inference.py when imported by Streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# The app will use the first model file that exists in this list.
# Keep the final Notebook 2 model first.
MODEL_CANDIDATES = [
    MODELS_DIR / "notebook2_gradient_boosting.joblib",
    MODELS_DIR / "notebook2_random_forest.joblib",
    MODELS_DIR / "optimized_extra_trees.joblib",
    MODELS_DIR / "optimized_random_forest.joblib",
    MODELS_DIR / "baseline_random_forest.joblib",
    MODELS_DIR / "baseline_linear_regression.joblib",
]

# Optional metadata files. These are helpful but not required.
METADATA_CANDIDATES = [
    MODELS_DIR / "notebook2_gradient_boosting_metadata.json",
    MODELS_DIR / "notebook2_model_metadata.json",
    MODELS_DIR / "model_metadata.json",
]

# Exact input columns expected by the project notebooks and Streamlit app.
REQUIRED_FEATURE_COLUMNS = [
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

NUMERIC_FEATURES = [
    "tubercle_amplitude",
    "tubercle_wavelength",
    "root_chord",
    "tip_chord",
    "sweep_angle",
    "angle_of_attack",
    "airspeed",
]

CATEGORICAL_FEATURES = ["airfoil_family", "tubercle_shape"]

VALID_AIRFOIL_FAMILIES = {"symmetric", "cambered", "biomimetic"}
VALID_TUBERCLE_SHAPES = {"none", "whale", "biomimetic_v1"}

_cached_model: Any = None
_cached_model_path: Optional[Path] = None
_cached_metadata: Optional[Dict[str, Any]] = None
_cached_metadata_path: Optional[Path] = None


# -----------------------------------------------------------------------------
# Model and metadata loading
# -----------------------------------------------------------------------------
def get_required_feature_columns() -> List[str]:
    """Return the exact feature columns expected by the trained pipeline."""
    return REQUIRED_FEATURE_COLUMNS.copy()


def list_available_models() -> List[str]:
    """Return saved model filenames currently found in the models/ folder."""
    if not MODELS_DIR.exists():
        return []
    return sorted(path.name for path in MODELS_DIR.glob("*.joblib"))


def _find_model_path(explicit_model_filename: Optional[str] = None) -> Path:
    """Find the model file to load."""
    if explicit_model_filename:
        requested_path = MODELS_DIR / explicit_model_filename
        if requested_path.exists():
            return requested_path
        raise FileNotFoundError(
            f"Requested model '{explicit_model_filename}' was not found in {MODELS_DIR}. "
            f"Available .joblib files: {list_available_models()}"
        )

    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return model_path

    raise FileNotFoundError(
        "No saved model file found in the models/ directory. Expected one of: "
        + ", ".join(path.name for path in MODEL_CANDIDATES)
        + f". Available .joblib files: {list_available_models()}"
    )


def load_model(
    force_reload: bool = False,
    explicit_model_filename: Optional[str] = None,
) -> Any:
    """Load and cache the saved scikit-learn model pipeline."""
    global _cached_model, _cached_model_path

    model_path = _find_model_path(explicit_model_filename=explicit_model_filename)

    if force_reload or _cached_model is None or _cached_model_path != model_path:
        _cached_model = joblib.load(model_path)
        _cached_model_path = model_path

    return _cached_model


def get_selected_model_path(explicit_model_filename: Optional[str] = None) -> Path:
    """Return the model path that inference.py will currently use."""
    return _find_model_path(explicit_model_filename=explicit_model_filename)


def _find_metadata_path() -> Optional[Path]:
    """Find an optional model metadata JSON file, if one exists."""
    for metadata_path in METADATA_CANDIDATES:
        if metadata_path.exists():
            return metadata_path
    return None


def load_metadata(force_reload: bool = False) -> Optional[Dict[str, Any]]:
    """Load optional JSON metadata saved by the notebook."""
    global _cached_metadata, _cached_metadata_path

    metadata_path = _find_metadata_path()
    if metadata_path is None:
        return None

    if force_reload or _cached_metadata is None or _cached_metadata_path != metadata_path:
        with metadata_path.open("r", encoding="utf-8") as f:
            _cached_metadata = json.load(f)
        _cached_metadata_path = metadata_path

    return _cached_metadata


def get_model_feature_columns(model: Optional[Any] = None) -> List[str]:
    """Return model feature columns when available, otherwise use project defaults."""
    if model is None:
        model = load_model()

    if hasattr(model, "feature_names_in_"):
        return [str(col) for col in model.feature_names_in_]

    metadata = load_metadata()
    if metadata:
        for key in ["feature_columns", "features", "input_columns"]:
            if key in metadata and isinstance(metadata[key], list):
                return [str(col) for col in metadata[key]]

    return get_required_feature_columns()


def get_model_status(explicit_model_filename: Optional[str] = None) -> Dict[str, Any]:
    """Return useful diagnostics for the Streamlit app."""
    model = load_model(explicit_model_filename=explicit_model_filename)
    model_path = get_selected_model_path(explicit_model_filename=explicit_model_filename)
    metadata_path = _find_metadata_path()

    final_estimator = model
    if hasattr(model, "steps") and model.steps:
        final_estimator = model.steps[-1][1]

    return {
        "project_root": str(PROJECT_ROOT),
        "models_dir": str(MODELS_DIR),
        "selected_model_file": str(model_path.relative_to(PROJECT_ROOT)),
        "available_model_files": list_available_models(),
        "selected_metadata_file": str(metadata_path.relative_to(PROJECT_ROOT)) if metadata_path else None,
        "model_type": type(model).__name__,
        "final_estimator_type": type(final_estimator).__name__,
        "model_feature_columns": get_model_feature_columns(model),
        "required_app_columns": get_required_feature_columns(),
    }


# -----------------------------------------------------------------------------
# Input validation and prediction
# -----------------------------------------------------------------------------
def _normalize_string(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")
    normalized = str(value).strip().lower()
    if not normalized:
        raise ValueError(f"Field '{field_name}' cannot be empty")
    return normalized


def _coerce_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for '{field_name}': {value}") from exc


def validate_input_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a one-row input dictionary from Streamlit."""
    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in input_dict]
    if missing:
        raise ValueError("Missing required input fields: " + ", ".join(missing))

    cleaned: Dict[str, Any] = {
        "airfoil_family": _normalize_string(input_dict["airfoil_family"], "airfoil_family"),
        "tubercle_amplitude": _coerce_float(input_dict["tubercle_amplitude"], "tubercle_amplitude"),
        "tubercle_wavelength": _coerce_float(input_dict["tubercle_wavelength"], "tubercle_wavelength"),
        "tubercle_shape": _normalize_string(input_dict["tubercle_shape"], "tubercle_shape"),
        "root_chord": _coerce_float(input_dict["root_chord"], "root_chord"),
        "tip_chord": _coerce_float(input_dict["tip_chord"], "tip_chord"),
        "sweep_angle": _coerce_float(input_dict["sweep_angle"], "sweep_angle"),
        "angle_of_attack": _coerce_float(input_dict["angle_of_attack"], "angle_of_attack"),
        "airspeed": _coerce_float(input_dict["airspeed"], "airspeed"),
    }

    if cleaned["airfoil_family"] not in VALID_AIRFOIL_FAMILIES:
        raise ValueError(
            f"Invalid airfoil_family '{cleaned['airfoil_family']}'. "
            f"Expected one of: {sorted(VALID_AIRFOIL_FAMILIES)}"
        )

    if cleaned["tubercle_shape"] not in VALID_TUBERCLE_SHAPES:
        raise ValueError(
            f"Invalid tubercle_shape '{cleaned['tubercle_shape']}'. "
            f"Expected one of: {sorted(VALID_TUBERCLE_SHAPES)}"
        )

    if cleaned["airfoil_family"] != "biomimetic":
        # Keep non-biomimetic cases physically consistent.
        cleaned["tubercle_amplitude"] = 0.0
        cleaned["tubercle_wavelength"] = 0.0
        cleaned["tubercle_shape"] = "none"

    if cleaned["tubercle_amplitude"] < 0:
        raise ValueError("tubercle_amplitude cannot be negative")
    if cleaned["tubercle_wavelength"] < 0:
        raise ValueError("tubercle_wavelength cannot be negative")
    if cleaned["root_chord"] <= 0:
        raise ValueError("root_chord must be greater than 0")
    if cleaned["tip_chord"] <= 0:
        raise ValueError("tip_chord must be greater than 0")
    if cleaned["airspeed"] <= 0:
        raise ValueError("airspeed must be greater than 0")

    return cleaned


def preprocess_inputs(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Convert one input dictionary into a one-row DataFrame."""
    cleaned_input = validate_input_dict(input_dict)
    return pd.DataFrame([cleaned_input], columns=REQUIRED_FEATURE_COLUMNS)


def predict_separation(
    processed_input: pd.DataFrame,
    explicit_model_filename: Optional[str] = None,
    force_reload: bool = False,
) -> float:
    """Run model inference and return predicted separation_x_over_c."""
    if not isinstance(processed_input, pd.DataFrame):
        raise TypeError("processed_input must be a pandas DataFrame")

    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in processed_input.columns]
    if missing:
        raise ValueError("Processed input is missing required columns: " + ", ".join(missing))

    model = load_model(force_reload=force_reload, explicit_model_filename=explicit_model_filename)
    prediction = float(model.predict(processed_input[REQUIRED_FEATURE_COLUMNS])[0])

    # separation_x_over_c is physically normalized to [0, 1].
    return float(np.clip(prediction, 0.0, 1.0))


def predict_from_dict(
    input_dict: Dict[str, Any],
    explicit_model_filename: Optional[str] = None,
    force_reload: bool = False,
) -> float:
    """Validate input, build the DataFrame, and return one prediction."""
    processed_input = preprocess_inputs(input_dict)
    return predict_separation(
        processed_input,
        explicit_model_filename=explicit_model_filename,
        force_reload=force_reload,
    )


def predict_batch_from_dicts(
    input_dicts: Iterable[Dict[str, Any]],
    explicit_model_filename: Optional[str] = None,
) -> pd.DataFrame:
    """Run predictions for many input dictionaries and return a DataFrame."""
    cleaned_rows = [validate_input_dict(row) for row in input_dicts]
    df = pd.DataFrame(cleaned_rows, columns=REQUIRED_FEATURE_COLUMNS)
    model = load_model(explicit_model_filename=explicit_model_filename)
    predictions = model.predict(df[REQUIRED_FEATURE_COLUMNS])
    df["predicted_separation_x_over_c"] = np.clip(predictions.astype(float), 0.0, 1.0)
    return df


# -----------------------------------------------------------------------------
# Sensitivity helpers for the Streamlit app
# -----------------------------------------------------------------------------
def make_sensitivity_sweep(
    input_dict: Dict[str, Any],
    variable: str = "angle_of_attack",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    step: float = 0.25,
    explicit_model_filename: Optional[str] = None,
) -> pd.DataFrame:
    """Create a one-variable sweep around the current input.

    This is especially useful for tree-based models, whose predictions can be
    piecewise constant. The sweep shows whether the current local region is flat
    or whether a larger change in angle, airspeed, or geometry changes output.
    """
    if variable not in NUMERIC_FEATURES:
        raise ValueError(f"Sensitivity variable must be numeric. Got: {variable}")
    if step <= 0:
        raise ValueError("step must be positive")

    cleaned = validate_input_dict(input_dict)
    current_value = float(cleaned[variable])

    if min_value is None:
        min_value = current_value - 5.0
    if max_value is None:
        max_value = current_value + 5.0
    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")

    values = np.arange(float(min_value), float(max_value) + (step / 2.0), float(step))
    rows: List[Dict[str, Any]] = []
    for value in values:
        row = cleaned.copy()
        row[variable] = float(value)
        rows.append(row)

    sweep_df = predict_batch_from_dicts(rows, explicit_model_filename=explicit_model_filename)
    sweep_df["sweep_variable"] = variable
    sweep_df["sweep_value"] = sweep_df[variable].astype(float)
    sweep_df["is_current_input_nearest"] = np.isclose(
        sweep_df["sweep_value"], current_value, atol=step / 2.0
    )
    return sweep_df


def local_delta_report(
    input_dict: Dict[str, Any],
    variable: str = "angle_of_attack",
    delta: float = 1.0,
    explicit_model_filename: Optional[str] = None,
) -> Dict[str, float]:
    """Compare prediction at current input, current - delta, and current + delta."""
    if variable not in NUMERIC_FEATURES:
        raise ValueError(f"Delta variable must be numeric. Got: {variable}")
    if delta <= 0:
        raise ValueError("delta must be positive")

    cleaned = validate_input_dict(input_dict)
    current_value = float(cleaned[variable])

    lower_input = cleaned.copy()
    upper_input = cleaned.copy()
    lower_input[variable] = current_value - delta
    upper_input[variable] = current_value + delta

    current_prediction = predict_from_dict(cleaned, explicit_model_filename=explicit_model_filename)
    lower_prediction = predict_from_dict(lower_input, explicit_model_filename=explicit_model_filename)
    upper_prediction = predict_from_dict(upper_input, explicit_model_filename=explicit_model_filename)

    return {
        "variable": variable,
        "current_value": current_value,
        "delta": float(delta),
        "lower_value": current_value - delta,
        "upper_value": current_value + delta,
        "lower_prediction": lower_prediction,
        "current_prediction": current_prediction,
        "upper_prediction": upper_prediction,
        "change_from_lower_to_current": current_prediction - lower_prediction,
        "change_from_current_to_upper": upper_prediction - current_prediction,
    }


def describe_model_response(delta_report: Dict[str, float], tolerance: float = 0.001) -> str:
    """Explain whether the model is changing locally or staying flat."""
    lower_change = abs(float(delta_report["change_from_lower_to_current"]))
    upper_change = abs(float(delta_report["change_from_current_to_upper"]))

    if lower_change < tolerance and upper_change < tolerance:
        return (
            "The prediction is locally flat for this small input change. This is normal for "
            "tree-based models because they split inputs into regions. It also means the "
            "dataset may need more simulation rows near this exact input range to resolve "
            "smaller one-degree differences."
        )

    return (
        "The prediction changes locally for this input. The model is responding to the "
        "selected variable in this region of the design space."
    )


def describe_prediction(separation_x_over_c: float) -> str:
    """Convert the numerical prediction into a simple risk interpretation."""
    if separation_x_over_c < 0.40:
        return "High separation risk"
    if separation_x_over_c < 0.70:
        return "Moderate separation risk"
    return "Low separation risk"


if __name__ == "__main__":
    example_input = {
        "airfoil_family": "biomimetic",
        "tubercle_amplitude": 28.537,
        "tubercle_wavelength": 49.607,
        "tubercle_shape": "whale",
        "root_chord": 1.0,
        "tip_chord": 1.0,
        "sweep_angle": 0.0,
        "angle_of_attack": 18.0,
        "airspeed": 30.0,
    }

    print(get_model_status())
    print("Prediction:", predict_from_dict(example_input))
    print("Delta report:", local_delta_report(example_input))
