"""Inference utilities for the Aero Toolkit Streamlit app.

This file is intentionally conservative: the app loads a saved scikit-learn
pipeline/model from the models/ directory and then makes one prediction from
an input dictionary. Keep the feature names here synchronized with Notebook 2
and app/app.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd

# Robust project-root discovery for both local runs and Streamlit Cloud.
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# Keep a stable production alias first. This lets the training notebook change
# algorithms without breaking the app. If the final model is Extra Trees, save
# or copy it as models/production_model.joblib as well.
MODEL_CANDIDATES = [
    MODELS_DIR / "production_model.joblib",
    MODELS_DIR / "notebook2_extra_trees.joblib",
    MODELS_DIR / "notebook2_gradient_boosting.joblib",
    MODELS_DIR / "notebook2_random_forest.joblib",
    MODELS_DIR / "optimized_extra_trees.joblib",
    MODELS_DIR / "optimized_random_forest.joblib",
    MODELS_DIR / "baseline_random_forest.joblib",
    MODELS_DIR / "baseline_linear_regression.joblib",
]

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

# -----------------------------------------------------------------------------
# Physical bounds on the target
# -----------------------------------------------------------------------------
# separation_x_over_c is a normalized chord position, so anything outside
# [0, 1] is physically meaningless. The regressor is free to produce such
# values under extrapolation, so predictions are clipped before display.
#
# Clipping is a *safety net*, not a result: a clipped value tells you the model
# left its supported range, and it should not be presented with the same
# confidence as an interior prediction. predict_raw_from_dict() below exposes
# the unclipped output so callers can detect and report this.
SEPARATION_MIN = 0.0
SEPARATION_MAX = 1.0

# -----------------------------------------------------------------------------
# Model evaluation metric (R^2)
# -----------------------------------------------------------------------------
# The held-out R^2 is computed at *training* time and is NOT stored inside the
# saved estimator, so it has to be supplied to the app separately. Two options:
#   1) (recommended) Have Notebook 2 write a metrics file next to the model, e.g.
#          models/metrics.json -> {"r2": 0.93}
#      or, if you keep several models around, key it by filename:
#          {"production_model.joblib": {"r2": 0.93}}
#   2) Paste the number into DEFAULT_MODEL_R2 below for a quick hard-coded value.
METRICS_CANDIDATES = [
    MODELS_DIR / "metrics.json",
    MODELS_DIR / "model_metrics.json",
    MODELS_DIR / "notebook2_metrics.json",
]

# Quick fallback if you don't want to manage a metrics file. Set to e.g. 0.93.
DEFAULT_MODEL_R2: float | None = None

# Accepted JSON keys for the R^2 value, in priority order.
_R2_KEYS = ("r2", "test_r2", "r2_score", "R2", "r_squared")

_CACHED_MODEL: Any | None = None
_CACHED_MODEL_PATH: Path | None = None


def get_required_feature_columns() -> List[str]:
    """Return the exact feature order expected by the trained model."""
    return list(REQUIRED_FEATURE_COLUMNS)


def _available_joblib_files() -> List[str]:
    """Return available model files for diagnostics."""
    if not MODELS_DIR.exists():
        return []
    return sorted(path.name for path in MODELS_DIR.glob("*.joblib"))


def find_model_path() -> Path:
    """Find the first supported model file in the models directory."""
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate

    expected = ", ".join(path.name for path in MODEL_CANDIDATES)
    available = _available_joblib_files()
    raise FileNotFoundError(
        "No saved model file found in the models/ directory. "
        f"Expected one of: {expected}. Available .joblib files: {available}"
    )


def load_model() -> Any:
    """Load and cache the saved model/pipeline."""
    global _CACHED_MODEL, _CACHED_MODEL_PATH

    model_path = find_model_path()
    if _CACHED_MODEL is None or _CACHED_MODEL_PATH != model_path:
        _CACHED_MODEL = joblib.load(model_path)
        _CACHED_MODEL_PATH = model_path
    return _CACHED_MODEL


def get_loaded_model_path() -> str:
    """Return the filename/path of the model currently selected by inference."""
    return str(find_model_path())


def _read_metrics_file() -> Dict[str, Any]:
    """Return the contents of the first readable metrics JSON, or {} if none."""
    for candidate in METRICS_CANDIDATES:
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, OSError):
                continue
    return {}


def get_model_r2_score() -> float | None:
    """Return the trained model's held-out R^2 score, or None if unknown.

    Resolution order:
      1) a metrics JSON in models/ keyed by the selected model filename
         (e.g. {"production_model.joblib": {"r2": 0.93}});
      2) a top-level value in that same JSON (e.g. {"r2": 0.93});
      3) the DEFAULT_MODEL_R2 constant above;
      4) None.

    Returning None lets the app display "N/A" rather than a fabricated number.
    """
    metrics = _read_metrics_file()

    try:
        model_name: str | None = Path(get_loaded_model_path()).name
    except Exception:
        model_name = None

    # Sources to inspect, most specific first.
    sources: List[Any] = []
    if model_name and isinstance(metrics.get(model_name), dict):
        sources.append(metrics[model_name])
    sources.append(metrics)  # allow a flat {"r2": 0.93}

    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in _R2_KEYS:
            if key in source:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    pass

    return DEFAULT_MODEL_R2


def validate_input(input_dict: Dict[str, Any]) -> None:
    """Check that the app provided every feature required by the model."""
    missing = [column for column in REQUIRED_FEATURE_COLUMNS if column not in input_dict]
    if missing:
        raise ValueError(f"Missing required model input columns: {missing}")


def make_input_dataframe(input_dict: Dict[str, Any]) -> pd.DataFrame:
    """Convert a single app input dictionary into a one-row dataframe."""
    validate_input(input_dict)
    row = {column: input_dict[column] for column in REQUIRED_FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=REQUIRED_FEATURE_COLUMNS)


def clip_prediction(value: float) -> float:
    """Clamp a raw prediction into the physical range [0, 1]."""
    return max(SEPARATION_MIN, min(SEPARATION_MAX, value))


def predict_raw_from_dict(input_dict: Dict[str, Any]) -> float:
    """Run the saved model and return the UNCLIPPED float prediction.

    This is the value the model actually produced. It may fall outside [0, 1]
    when the inputs push the model to extrapolate. Callers that display results
    to a user should prefer this function together with clip_prediction(), so
    that out-of-range outputs can be reported rather than silently hidden.
    """
    model = load_model()
    input_df = make_input_dataframe(input_dict)
    prediction = model.predict(input_df)
    return float(prediction[0])


def predict_with_clipping(input_dict: Dict[str, Any]) -> Tuple[float, float, bool]:
    """Return (raw_value, clipped_value, was_clipped) for one input dictionary."""
    raw_value = predict_raw_from_dict(input_dict)
    clipped_value = clip_prediction(raw_value)
    return raw_value, clipped_value, raw_value != clipped_value


def predict_from_dict(input_dict: Dict[str, Any]) -> float:
    """Run the saved model on one input dictionary and return one float prediction.

    The returned value is clipped to [0, 1]. Callers that need to know whether
    clipping occurred should use predict_with_clipping() or predict_raw_from_dict()
    instead; this function is kept for backward compatibility and silently hides
    the distinction between a genuine 1.0 and a clamped 1.4.
    """
    return clip_prediction(predict_raw_from_dict(input_dict))


def describe_prediction(separation_x_over_c: float) -> str:
    """Convert a numeric separation prediction into a workshop-friendly label."""
    if separation_x_over_c >= 0.80:
        return "Delayed separation / promising design"
    if separation_x_over_c >= 0.60:
        return "Moderate separation behavior"
    return "Early separation / redesign recommended"


if __name__ == "__main__":
    print("Python executable:", sys.executable)
    print("Project root:", PROJECT_ROOT)
    print("Models directory:", MODELS_DIR)
    print("Available model files:", _available_joblib_files())
    print("Selected model:", find_model_path())
    print("Model R^2:", get_model_r2_score())
