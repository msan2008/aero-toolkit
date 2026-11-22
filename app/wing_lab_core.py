"""Wing Lab challenge logic, kept separate from the Streamlit UI.

Nothing here trains, alters, or re-implements the model. Every prediction goes
through src.inference.predict_with_clipping(), the same backend Advanced Mode
uses. This module only:
  * fixes the challenge's flight conditions,
  * restricts tubercle inputs to geometries that exist in the training data,
  * builds the exact input dictionary the model expects,
  * compares an attempt to the starting design under identical conditions,
  * formats results for download.

It does not import streamlit, so it can be unit-tested directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from src.inference import get_model_metrics, get_required_feature_columns, predict_with_clipping

# -----------------------------------------------------------------------------
# Model-supported inputs
# -----------------------------------------------------------------------------
# The Phase 2 OpenFOAM dataset (data/processed/openfoam_phase2_cleaned.csv)
# contains exactly six biomimetic tubercle geometries, all with shape "whale".
# In that dataset amplitude and wavelength are paired: each amplitude appears
# with exactly one wavelength. Letting students set them independently would
# produce combinations the model never saw, so Wing Lab offers these six pairs
# only. Values are copied verbatim from the dataset (millimetres).
TUBERCLE_GEOMETRIES: List[Dict[str, float]] = [
    {"tubercle_amplitude": 26.247, "tubercle_wavelength": 49.607},
    {"tubercle_amplitude": 28.537, "tubercle_wavelength": 47.334},
    {"tubercle_amplitude": 29.663, "tubercle_wavelength": 46.344},
    {"tubercle_amplitude": 30.421, "tubercle_wavelength": 45.873},
    {"tubercle_amplitude": 31.564, "tubercle_wavelength": 43.639},
    {"tubercle_amplitude": 32.734, "tubercle_wavelength": 42.337},
]

# Fixed for the whole challenge so that every comparison is controlled.
# Both values are points on the training grid (AoA 0-24 deg in 1 deg steps,
# airspeed 15 or 30 m/s). Geometry is held at the model's constant values.
CHALLENGE_CONDITIONS: Dict[str, Any] = {
    "airfoil_family": "biomimetic",
    "tubercle_shape": "whale",
    "angle_of_attack": 12.0,
    "airspeed": 15,
    "root_chord": 1.0,
    "tip_chord": 1.0,
    "sweep_angle": 0.0,
}

STARTING_GEOMETRY_INDEX = 0  # the 26.2 mm / 49.6 mm preset
MAX_ATTEMPTS = 3

# Used only if models/metrics.json cannot be read. It matches the value
# recorded there (holdout RMSE, in units of x/c).
FALLBACK_RMSE = 0.0203


def geometry_label(index: int) -> str:
    g = TUBERCLE_GEOMETRIES[index]
    return (
        f"Design {index + 1}: amplitude {g['tubercle_amplitude']:.1f} mm · "
        f"wavelength {g['tubercle_wavelength']:.1f} mm"
    )


def build_payload(geometry_index: int) -> Dict[str, Any]:
    """Return the exact model input dict for one supported geometry."""
    if not 0 <= geometry_index < len(TUBERCLE_GEOMETRIES):
        raise ValueError(f"Unsupported tubercle geometry index: {geometry_index}")
    payload = {**CHALLENGE_CONDITIONS, **TUBERCLE_GEOMETRIES[geometry_index]}
    missing = set(get_required_feature_columns()) - set(payload)
    if missing:  # defensive: the schema lives in src/inference.py
        raise ValueError(f"Wing Lab payload is missing model features: {sorted(missing)}")
    return payload


def run_design(geometry_index: int) -> Dict[str, Any]:
    """Predict one design using the existing backend. Returns a result record."""
    payload = build_payload(geometry_index)
    raw, clipped, was_clipped = predict_with_clipping(payload)
    return {
        "geometry_index": geometry_index,
        "inputs": payload,
        "raw": float(raw),
        "x_over_c": float(clipped),
        "was_clipped": bool(was_clipped),
    }


def comparison_threshold() -> float:
    """Differences smaller than the model's holdout RMSE are not distinguishable."""
    try:
        return float(get_model_metrics().get("rmse", FALLBACK_RMSE))
    except Exception:
        return FALLBACK_RMSE


def compare_to_start(attempt_x: float, start_x: float, threshold: float) -> Dict[str, Any]:
    """Classify the difference between an attempt and the starting design."""
    delta = attempt_x - start_x
    if abs(delta) <= threshold:
        verdict = "similar"
    elif delta > 0:
        verdict = "later"
    else:
        verdict = "earlier"
    return {"delta": delta, "verdict": verdict}


def explain_comparison(verdict: str, delta: float, threshold: float) -> str:
    """Plain-language meaning of a comparison. It never claims a wing is better."""
    pct = abs(delta) * 100
    if verdict == "similar":
        return (
            f"The difference ({delta:+.3f}) is smaller than the model's typical error "
            f"(about ±{threshold:.2f} x/c), so the model can't really tell these two designs "
            "apart. Treat them as about the same."
        )
    if verdict == "later":
        return (
            f"The model predicts the airflow stays attached about {pct:.0f}% of the chord "
            "further back than on the starting design before it separates."
        )
    return (
        f"The model predicts the airflow separates about {pct:.0f}% of the chord earlier "
        "than on the starting design, so less of the upper surface keeps attached flow."
    )


def results_dataframe(start: Dict[str, Any] | None, attempts: List[Dict[str, Any]]) -> pd.DataFrame:
    """One row per design (start + attempts) for CSV download."""
    threshold = comparison_threshold()
    rows = []
    designs = ([("starting design", start)] if start else []) + [
        (f"attempt {i}", a) for i, a in enumerate(attempts, start=1)
    ]
    for name, rec in designs:
        cmp = compare_to_start(rec["x_over_c"], start["x_over_c"], threshold) if start else None
        rows.append({
            "design": name,
            **rec["inputs"],
            "predicted_separation_x_over_c": round(rec["x_over_c"], 4),
            "raw_model_output": round(rec["raw"], 4),
            "clipped": rec["was_clipped"],
            "delta_vs_start": round(cmp["delta"], 4) if cmp else None,
            "comparison": cmp["verdict"] if cmp else None,
        })
    df = pd.DataFrame(rows)
    df["note"] = (
        "ML screening prediction, not a CFD result. Does not predict lift, drag, or efficiency."
    )
    return df
