from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

# Project paths
PROJECT_ROOT = Path("models/notebook2_gradient_boosting.joblib").resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Update this list if the final saved model filename changes.
MODEL_CANDIDATES = [
    #MODELS_DIR / "optimized_random_forest.joblib",
    #MODELS_DIR / "optimized_extra_trees.joblib",
    #MODELS_DIR / "baseline_random_forest.joblib",
    #MODELS_DIR / "baseline_linear_regression.joblib",
    MODELS_DIR / "notebook2_gradient_boosting.joblib",
]

# Exact feature columns expected by the trained model pipeline.
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

VALID_AIRFOIL_FAMILIES = {"symmetric", "cambered", "biomimetic"}
VALID_TUBERCLE_SHAPES = {"none", "whale", "biomimetic_v1"}

_cached_model = None
_cached_model_path = None


def get_required_feature_columns() -> List[str]:
    """
    Return the exact feature column order expected by the trained model.
    """
    return REQUIRED_FEATURE_COLUMNS.copy()


def _find_model_path() -> Path:
    """
    Return the first available saved model path from the candidate list.
    """
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return model_path

    raise FileNotFoundError(
        "No saved model file found in the models/ directory. "
        "Expected one of: "
        + ", ".join(path.name for path in MODEL_CANDIDATES)
    )


def load_model(force_reload: bool = False):
    """
    Load and cache the first available saved model pipeline.

    The saved artifact should be the full sklearn pipeline, not just the raw model,
    so preprocessing is preserved automatically.
    """
    global _cached_model, _cached_model_path

    model_path = _find_model_path()

    if force_reload or _cached_model is None or _cached_model_path != model_path:
        _cached_model = joblib.load(model_path)
        _cached_model_path = model_path

    return _cached_model


def _normalize_string(value: Any, field_name: str) -> str:
    """
    Normalize a string input by stripping whitespace and lowercasing.
    """
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")

    normalized = str(value).strip().lower()
    if normalized == "":
        raise ValueError(f"Field '{field_name}' cannot be empty")

    return normalized


def _coerce_float(value: Any, field_name: str) -> float:
    """
    Convert a value to float and raise a clear error if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{field_name}': {value}") from exc


def validate_input_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the input dictionary contains all required fields and
    normalize the values into the exact format expected by the model.
    """
    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in input_dict]
    if missing:
        raise ValueError(
            "Missing required input fields: " + ", ".join(missing)
        )

    cleaned = {
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
    """
    Convert the validated input dictionary into a one-row DataFrame with
    the exact feature column order expected by the trained pipeline.

    The actual preprocessing is handled by the saved sklearn pipeline.
    """
    cleaned_input = validate_input_dict(input_dict)
    input_df = pd.DataFrame([cleaned_input], columns=REQUIRED_FEATURE_COLUMNS)
    return input_df


def predict_separation(processed_input: pd.DataFrame) -> float:
    """
    Run model inference and return predicted separation_x_over_c.

    Output is clipped to the physically meaningful range [0, 1].
    """
    if not isinstance(processed_input, pd.DataFrame):
        raise TypeError("processed_input must be a pandas DataFrame")

    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in processed_input.columns]
    if missing:
        raise ValueError(
            "Processed input is missing required feature columns: " + ", ".join(missing)
        )

    model = load_model()
    prediction = float(model.predict(processed_input[REQUIRED_FEATURE_COLUMNS])[0])

    # Keep output in a sensible physical range.
    prediction = max(0.0, min(1.0, prediction))
    return prediction


def predict_from_dict(input_dict: Dict[str, Any]) -> float:
    """
    Convenience wrapper for Streamlit.

    App flow:
    1. collect widget values into input_dict
    2. call predict_from_dict(input_dict)
    3. display the prediction in the app
    """
    processed_input = preprocess_inputs(input_dict)
    prediction = predict_separation(processed_input)
    return prediction


def describe_prediction(separation_x_over_c: float) -> str:
    """
    Convert the numerical prediction into a simple UI interpretation.

    These thresholds are a first-pass heuristic and can be updated later
    once the real data distribution is better established.
    """
    if separation_x_over_c < 0.40:
        return "High separation risk"
    elif separation_x_over_c < 0.70:
        return "Moderate separation risk"
    else:
        return "Low separation risk"


if __name__ == "__main__":
    example_input = {
        "airfoil_family": "biomimetic",
        "tubercle_amplitude": 26.247,
        "tubercle_wavelength": 49.607,
        "tubercle_shape": "whale",
        "root_chord": 1.0,
        "tip_chord": 1.0,
        "sweep_angle": 0.0,
        "angle_of_attack": 10.0,
        "airspeed": 30.0,
    }

    prediction = predict_from_dict(example_input)
    label = describe_prediction(prediction)

    print(f"Predicted separation_x_over_c: {prediction:.4f}")
    print(f"Interpretation: {label}")

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

# Project paths
PROJECT_ROOT = Path("models/notebook2_gradient_boosting.joblib").resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Update this list if the final saved model filename changes.
MODEL_CANDIDATES = [
    #MODELS_DIR / "optimized_random_forest.joblib",
    #MODELS_DIR / "optimized_extra_trees.joblib",
    #MODELS_DIR / "baseline_random_forest.joblib",
    MODELS_DIR / "notebook2_gradient_boosting.joblib",
]

# Exact feature columns expected by the trained model pipeline.
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

VALID_AIRFOIL_FAMILIES = {"symmetric", "cambered", "biomimetic"}
VALID_TUBERCLE_SHAPES = {"none", "whale", "biomimetic_v1"}

_cached_model = None
_cached_model_path = None


def get_required_feature_columns() -> List[str]:
    """
    Return the exact feature column order expected by the trained model.
    """
    return REQUIRED_FEATURE_COLUMNS.copy()


def _find_model_path(explicit_model_filename: Optional[str] = None) -> Path:
    """
    Return the first available saved model path from the candidate list.
    """
    if explicit_model_filename:
         explicit_path = MODELS_DIR / explicit_model_filename
         if explicit_path.exists():
            return model_path
         raise FileNotFoundError(
          f"Requested model '{explicit_model_filename}' was not found in {MODELS_DIR}"
    )
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
          return model_path

    raise FileNotFoundError(
        "No saved model file found in the models/ directory. "
        "Expected one of: " + ", ".join(path.name for path in MODEL_CANDIDATES)
    )

def load_model(force_reload: bool = False, explicit_model_filename: Optional[str] = None):
    """
    Load and cache the first available saved model pipeline.

    The saved artifact should be the full sklearn pipeline, not just the raw model,
    so preprocessing is preserved automatically.
    """
    global _cached_model, _cached_model_path

    model_path = _find_model_path(explicit_model_filename = explicit_model_filename)

    if force_reload or _cached_model is None or _cached_model_path != model_path:
        _cached_model = joblib.load(model_path)
        _cached_model_path = model_path

    return _cached_model


def _normalize_string(value: Any, field_name: str) -> str:
    """
    Normalize a string input by stripping whitespace and lowercasing.
    """
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")

    normalized = str(value).strip().lower()
    if normalized == "":
        raise ValueError(f"Field '{field_name}' cannot be empty")

    return normalized


def _coerce_float(value: Any, field_name: str) -> float:
    """
    Convert a value to float and raise a clear error if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{field_name}': {value}") from exc


def validate_input_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the input dictionary contains all required fields and
    normalize the values into the exact format expected by the model.
    """
    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in input_dict]
    if missing:
        raise ValueError(
            "Missing required input fields: " + ", ".join(missing)
        )

    cleaned = {
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
    """
    Convert the validated input dictionary into a one-row DataFrame with
    the exact feature column order expected by the trained pipeline.

    The actual preprocessing is handled by the saved sklearn pipeline.
    """
    cleaned_input = validate_input_dict(input_dict)
    input_df = pd.DataFrame([cleaned_input], columns=REQUIRED_FEATURE_COLUMNS)
    return input_df


def predict_separation(processed_input: pd.DataFrame) -> float:
    """
    Run model inference and return predicted separation_x_over_c.

    Output is clipped to the physically meaningful range [0, 1].
    """
    if not isinstance(processed_input, pd.DataFrame):
        raise TypeError("processed_input must be a pandas DataFrame")

    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in processed_input.columns]
    if missing:
        raise ValueError(
            "Processed input is missing required feature columns: " + ", ".join(missing)
        )

    model = load_model(explicit_model_filename = explicit_mode_filename)
    prediction = float(model.predict(processed_input[REQUIRED_FEATURE_COLUMNS])[0])

    # Keep output in a sensible physical range.
    prediction = max(0.0, min(1.0, prediction))
    return prediction


def predict_from_dict(input_dict: Dict[str, Any]) -> float:
    """
    Convenience wrapper for Streamlit.

    App flow:
    1. collect widget values into input_dict
    2. call predict_from_dict(input_dict)
    3. display the prediction in the app
    """
    processed_input = preprocess_inputs(input_dict)
    prediction = predict_separation(processed_input, explicit_model_filename = explicit_model_filename)
    return prediction


def describe_prediction(separation_x_over_c: float) -> str:
    """
    Convert the numerical prediction into a simple UI interpretation.

    These thresholds are a first-pass heuristic and can be updated later
    once the real data distribution is better established.
    """
    if separation_x_over_c < 0.40:
        return "High separation risk"
    elif separation_x_over_c < 0.70:
        return "Moderate separation risk"
    else:
        return "Low separation risk"


if __name__ == "__main__":
    example_input = {
        "airfoil_family": "biomimetic",
        "tubercle_amplitude": 26.247,
        "tubercle_wavelength": 49.607,
        "tubercle_shape": "whale",
        "root_chord": 1.0,
        "tip_chord": 1.0,
        "sweep_angle": 0.0,
        "angle_of_attack": 10.0,
        "airspeed": 30.0,
    }

    prediction = predict_from_dict(example_input)
    label = describe_prediction(prediction)

    print(f"Predicted separation_x_over_c: {prediction:.4f}")
    print(f"Interpretation: {label}")

from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path("models/notebook2_gradient_boosting.joblib").resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Ordered from most preferred to least preferred. The code will load the first
# model file that exists in this list.
MODEL_CANDIDATES = [
    #MODELS_DIR / "optimized_extra_trees.joblib",
    #MODELS_DIR / "optimized_random_forest.joblib",
    #MODELS_DIR / "baseline_random_forest.joblib",
    MODELS_DIR / "notebook2_gradient_boosting.joblib",
]

# Exact feature schema expected by Notebook 1.5
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

# Valid category labels based on the current cleaned CFD dataset and app design
VALID_AIRFOIL_FAMILIES = {"symmetric", "cambered", "biomimetic"}
VALID_TUBERCLE_SHAPES = {"none", "whale", "biomimetic_v1"}

_cached_model = None
_cached_model_path = None


# -----------------------------------------------------------------------------
# Schema helpers
# -----------------------------------------------------------------------------
def get_required_feature_columns() -> List[str]:
    """
    Return the exact feature column order expected by the trained model pipeline.
    """
    return REQUIRED_FEATURE_COLUMNS.copy()


def _find_model_path(explicit_model_filename: Optional[str] = None) -> Path:
    """
    Return the first available saved model path.

    Parameters
    ----------
    explicit_model_filename : Optional[str]
        If provided, look for this exact file inside models/ first.
    """
    if explicit_model_filename:
        explicit_path = MODELS_DIR / explicit_model_filename
        if explicit_path.exists():
            return explicit_path
        raise FileNotFoundError(
            f"Requested model '{explicit_model_filename}' was not found in {MODELS_DIR}"
        )

    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return model_path

    raise FileNotFoundError(
        "No saved model file found in the models/ directory. "
        "Expected one of: " + ", ".join(path.name for path in MODEL_CANDIDATES)
    )


def load_model(force_reload: bool = False, explicit_model_filename: Optional[str] = None):
    """
    Load and cache the first available saved model pipeline.

    The saved artifact should be the full sklearn pipeline, not just the raw model,
    so preprocessing is preserved automatically.

    Parameters
    ----------
    force_reload : bool
        If True, reload the model from disk even if a cached model exists.
    explicit_model_filename : Optional[str]
        If provided, load this exact model filename from models/.
    """
    global _cached_model, _cached_model_path

    model_path = _find_model_path(explicit_model_filename=explicit_model_filename)

    if force_reload or _cached_model is None or _cached_model_path != model_path:
        _cached_model = joblib.load(model_path)
        _cached_model_path = model_path

    return _cached_model


# -----------------------------------------------------------------------------
# Input validation helpers
# -----------------------------------------------------------------------------
def _normalize_string(value: Any, field_name: str) -> str:
    """
    Normalize a string input by stripping whitespace and lowercasing.
    """
    if value is None:
        raise ValueError(f"Missing required field '{field_name}'")

    normalized = str(value).strip().lower()
    if normalized == "":
        raise ValueError(f"Field '{field_name}' cannot be empty")

    return normalized


def _coerce_float(value: Any, field_name: str) -> float:
    """
    Convert a value to float and raise a clear error if conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid value for '{field_name}': {value}") from exc


def validate_input_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the input dictionary contains all required fields and normalize
    the values into the exact format expected by the model.

    Returns
    -------
    Dict[str, Any]
        Cleaned dictionary with normalized strings and numeric floats.
    """
    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in input_dict]
    if missing:
        raise ValueError(
            "Missing required input fields: " + ", ".join(missing)
        )

    cleaned = {
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

    # Basic physical sanity checks appropriate for the current app design.
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

    # Sweep angle can be zero, but it should not be wildly invalid.
    if cleaned["sweep_angle"] < 0:
        raise ValueError("sweep_angle cannot be negative for the current V1 setup")

    return cleaned


# -----------------------------------------------------------------------------
# Preprocessing and prediction
# -----------------------------------------------------------------------------
def preprocess_inputs(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert the validated input dictionary into a one-row DataFrame with the exact
    feature column order expected by the trained pipeline.

    The actual preprocessing is handled by the saved sklearn pipeline.
    """
    cleaned_input = validate_input_dict(input_dict)
    input_df = pd.DataFrame([cleaned_input], columns=REQUIRED_FEATURE_COLUMNS)
    return input_df


def predict_separation(processed_input: pd.DataFrame, explicit_model_filename: Optional[str] = None) -> float:
    """
    Run model inference and return predicted separation_x_over_c.

    Parameters
    ----------
    processed_input : pd.DataFrame
        A one-row DataFrame with the exact feature schema.
    explicit_model_filename : Optional[str]
        If provided, load this exact model from models/.

    Returns
    -------
    float
        Predicted separation_x_over_c clipped to the physically meaningful [0, 1] range.
    """
    if not isinstance(processed_input, pd.DataFrame):
        raise TypeError("processed_input must be a pandas DataFrame")

    missing = [col for col in REQUIRED_FEATURE_COLUMNS if col not in processed_input.columns]
    if missing:
        raise ValueError(
            "Processed input is missing required feature columns: " + ", ".join(missing)
        )

    model = load_model(explicit_model_filename=explicit_model_filename)
    prediction = float(model.predict(processed_input[REQUIRED_FEATURE_COLUMNS])[0])

    # Keep output in a sensible physical range for x/c.
    prediction = max(0.0, min(1.0, prediction))
    return prediction


def predict_from_dict(input_dict: Dict[str, Any], explicit_model_filename: Optional[str] = None) -> float:
    """
    Convenience wrapper for Streamlit.

    App flow:
    1. collect widget values into input_dict
    2. call predict_from_dict(input_dict)
    3. display the prediction in the app
    """
    processed_input = preprocess_inputs(input_dict)
    prediction = predict_separation(processed_input, explicit_model_filename=explicit_model_filename)
    return prediction


def describe_prediction(separation_x_over_c: float) -> str:
    """
    Convert the numerical prediction into a simple UI interpretation.

    These thresholds are a first-pass heuristic and can be updated later once the
    real data distribution is better established.
    """
    if separation_x_over_c < 0.40:
        return "High separation risk"
    elif separation_x_over_c < 0.70:
        return "Moderate separation risk"
    else:
        return "Low separation risk"


# -----------------------------------------------------------------------------
# Simple command-line test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    example_input = {
        "airfoil_family": "biomimetic",
        "tubercle_amplitude": 26.247,
        "tubercle_wavelength": 49.607,
        "tubercle_shape": "whale",
        "root_chord": 1.0,
        "tip_chord": 1.0,
        "sweep_angle": 0.0,
        "angle_of_attack": 10.0,
        "airspeed": 30.0,
    }

    prediction = predict_from_dict(example_input)
    label = describe_prediction(prediction)

    print(f"Predicted separation_x_over_c: {prediction:.4f}")
    print(f"Interpretation: {label}")
