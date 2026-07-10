"""Inference utilities for the Aero Toolkit Streamlit app.

This file is intentionally conservative: the app loads a saved scikit-learn
pipeline/model from the models/ directory and then makes one prediction from
an input dictionary. Keep the feature names here synchronized with Notebook 2
and app/app.py.
"""

from __future__ import annotations

import json
import math
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
# Model evaluation metrics (R^2, MAE, RMSE)
# -----------------------------------------------------------------------------
# Held-out validation metrics are computed at *training* time and are NOT stored
# inside the saved estimator, so they have to be supplied to the app separately.
# Two options:
#   1) (recommended) Have Notebook 2 write a metrics file next to the model, e.g.
#          models/metrics.json -> {"r2": 0.93, "mae": 0.028, "rmse": 0.041}
#      or, if you keep several models around, key it by filename:
#          {"production_model.joblib": {"r2": 0.93, "mae": 0.028}}
#   2) Paste the numbers into DEFAULT_MODEL_METRICS below for quick hard-coded values.
#
# MAE and RMSE are in the units of the target, i.e. fractions of chord. An MAE of
# 0.028 means the model is typically off by about 2.8% of the chord length.
METRICS_CANDIDATES = [
    MODELS_DIR / "metrics.json",
    MODELS_DIR / "model_metrics.json",
    MODELS_DIR / "notebook2_metrics.json",
    PROJECT_ROOT / "metrics.json",
]


def _candidate_metrics_paths() -> List[Path]:
    """Explicit candidates first, then any other *metrics*.json in models/."""
    paths = list(METRICS_CANDIDATES)
    if MODELS_DIR.exists():
        for extra in sorted(MODELS_DIR.glob("*metrics*.json")):
            if extra not in paths:
                paths.append(extra)
    return paths


# Quick fallback if you don't want to manage a metrics file. Set to e.g. 0.93.
DEFAULT_MODEL_R2: float | None = None
DEFAULT_MODEL_MAE: float | None = None
DEFAULT_MODEL_RMSE: float | None = None

# Accepted JSON keys for each metric, in priority order. Held-out/test values are
# listed before generic ones so a file containing both prefers the test score.
_METRIC_ALIASES: Dict[str, Tuple[str, ...]] = {
    "r2": ("test_r2", "r2", "r2_score", "R2", "r_squared", "val_r2"),
    "mae": ("test_mae", "mae", "mean_absolute_error", "MAE", "val_mae"),
    "rmse": ("test_rmse", "rmse", "root_mean_squared_error", "RMSE", "val_rmse"),
    "mse": ("test_mse", "mse", "mean_squared_error", "MSE", "val_mse"),
}

# sklearn's cross_val_score reports errors negated (higher = better). If a file
# records those directly, read them and flip the sign back.
_NEGATED_ALIASES: Dict[str, Tuple[str, ...]] = {
    "mae": ("neg_mean_absolute_error", "test_neg_mean_absolute_error"),
    "mse": ("neg_mean_squared_error", "test_neg_mean_squared_error"),
    "rmse": ("neg_root_mean_squared_error", "test_neg_root_mean_squared_error"),
}

# Sub-dictionaries that commonly wrap the actual scores.
_NESTED_CONTAINER_KEYS = ("metrics", "scores", "test", "test_metrics", "evaluation", "results")

# Backwards compatibility: some callers import this directly.
_R2_KEYS = _METRIC_ALIASES["r2"]

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


def _read_metrics_file() -> Tuple[Dict[str, Any], Path | None]:
    """Return (parsed_metrics_dict, source_path), or ({}, None) if none readable.

    A top-level JSON list is accepted and its first dict element used, since
    some notebook exports write one record per model.
    """
    for candidate in _candidate_metrics_paths():
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            continue

        if isinstance(data, dict):
            return data, candidate
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    return item, candidate
    return {}, None


def _metric_sources() -> List[Dict[str, Any]]:
    """Return dicts to search for metrics, most specific first.

    Handles three layouts seen in practice:
        {"r2": 0.93}                                   flat
        {"production_model.joblib": {"r2": 0.93}}      keyed by model filename
        {"metrics": {"r2": 0.93}}                      wrapped in a container
    and any combination of the last two.
    """
    metrics, _ = _read_metrics_file()

    try:
        model_name: str | None = Path(get_loaded_model_path()).name
    except Exception:
        model_name = None

    sources: List[Dict[str, Any]] = []

    def add_with_nested(candidate: Any) -> None:
        if not isinstance(candidate, dict):
            return
        sources.append(candidate)
        for container_key in _NESTED_CONTAINER_KEYS:
            nested = candidate.get(container_key)
            if isinstance(nested, dict):
                sources.append(nested)

    if model_name:
        add_with_nested(metrics.get(model_name))
    add_with_nested(metrics)
    return sources


def _lookup_metric(name: str) -> float | None:
    """Find one metric by any accepted alias, including negated sklearn keys."""
    sources = _metric_sources()

    for source in sources:
        for key in _METRIC_ALIASES.get(name, ()):
            if key in source:
                try:
                    return float(source[key])
                except (TypeError, ValueError):
                    pass

    # Only fall back to negated scorers if no direct key was found.
    for source in sources:
        for key in _NEGATED_ALIASES.get(name, ()):
            if key in source:
                try:
                    return abs(float(source[key]))
                except (TypeError, ValueError):
                    pass
    return None


def get_metrics_diagnostics() -> Dict[str, Any]:
    """Explain exactly why metrics did or did not resolve.

    Intended for the app's diagnostics panel: when the UI shows "N/A", this
    says whether the file was missing, unreadable, or present-but-unrecognized.
    """
    searched = _candidate_metrics_paths()
    metrics, source_path = _read_metrics_file()

    try:
        model_name: str | None = Path(get_loaded_model_path()).name
    except Exception:
        model_name = None

    resolved = get_model_metrics()

    if source_path is None:
        reason = (
            "No metrics file found. Create one of the searched paths, or set "
            "DEFAULT_MODEL_R2 / DEFAULT_MODEL_MAE / DEFAULT_MODEL_RMSE in src/inference.py."
        )
    elif not resolved:
        reason = (
            f"Found {source_path.name}, but none of its keys matched a known metric name. "
            f"Top-level keys present: {sorted(metrics)[:12]}. Rename them to r2 / mae / rmse."
        )
    else:
        missing = [m for m in ("r2", "mae", "rmse") if m not in resolved]
        reason = f"Read {len(resolved)} metric(s) from {source_path.name}."
        if missing:
            reason += f" Not recorded in the file: {', '.join(missing)}."

    return {
        "searched_paths": [str(path) for path in searched],
        "existing_paths": [str(path) for path in searched if path.exists()],
        "source_file": str(source_path) if source_path else None,
        "selected_model": model_name,
        "top_level_keys": sorted(metrics) if metrics else [],
        "resolved_metrics": resolved,
        "explanation": reason,
    }


def get_model_metrics() -> Dict[str, float]:
    """Return whatever held-out validation metrics are available.

    Keys are a subset of {"r2", "mae", "rmse"}; absent metrics are omitted
    rather than faked, so the app can display only what it actually knows.

    Resolution order per metric:
      1) a metrics JSON in models/ keyed by the selected model filename;
      2) a top-level value in that same JSON;
      3) the corresponding DEFAULT_MODEL_* constant above.

    RMSE is derived from MSE when only MSE is recorded.
    """
    resolved: Dict[str, float] = {}

    r2 = _lookup_metric("r2")
    if r2 is None:
        r2 = DEFAULT_MODEL_R2
    if r2 is not None:
        resolved["r2"] = float(r2)

    mae = _lookup_metric("mae")
    if mae is None:
        mae = DEFAULT_MODEL_MAE
    if mae is not None:
        resolved["mae"] = float(mae)

    rmse = _lookup_metric("rmse")
    if rmse is None:
        mse = _lookup_metric("mse")
        # sqrt(MSE) is RMSE by definition; only derive it when RMSE is absent.
        if mse is not None and mse >= 0:
            rmse = math.sqrt(mse)
    if rmse is None:
        rmse = DEFAULT_MODEL_RMSE
    if rmse is not None:
        resolved["rmse"] = float(rmse)

    return resolved


def get_model_r2_score() -> float | None:
    """Return the trained model's held-out R^2 score, or None if unknown.

    Returning None lets the app display "N/A" rather than a fabricated number.
    Kept as a thin wrapper over get_model_metrics() for backward compatibility.
    """
    return get_model_metrics().get("r2", DEFAULT_MODEL_R2)


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
    print("Model metrics:", get_model_metrics())
