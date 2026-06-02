"""Inference utilities for the Aero Toolkit Streamlit app.

This file is intentionally conservative: the app loads a saved scikit-learn
pipeline/model from the models/ directory and then makes one prediction from
an input dictionary. Keep the feature names here synchronized with Notebook 2
and app/app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

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


def predict_from_dict(input_dict: Dict[str, Any]) -> float:
    """Run the saved model on one input dictionary and return one float prediction."""
    model = load_model()
    input_df = make_input_dataframe(input_dict)
    prediction = model.predict(input_df)
    value = float(prediction[0])

    # separation_x_over_c should be normalized. Clipping protects the UI from
    # rare extrapolation artifacts, but the training data should also be checked.
    return max(0.0, min(1.0, value))


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
