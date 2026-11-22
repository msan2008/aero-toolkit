"""Wing Lab game rules, kept separate from the Streamlit UI.

Two kinds of numbers live in this game, and they must never be confused:

  * MODEL (real): separation_x_over_c, predicted by the unchanged
    src.inference.predict_with_clipping() from model-supported inputs only.
  * GAME (invented): flight distance, height, points and unlocks. These are
    made-up game rules that turn x/c into something fun to watch. They are
    not physics and are labelled as such everywhere in the UI.

Game rule:  distance = MAINLAND_DISTANCE * launch_power * glide_efficiency
  * launch_power (0-1] comes from launch angle and speed only. It is capped at
    1.0, so no angle/speed combination can carry a wing across on its own.
  * glide_efficiency comes from the model's x/c and equals exactly 1.0 at
    WIN_X_OVER_C. Because launch_power <= 1, a wing can only reach the
    mainland if its predicted x/c >= WIN_X_OVER_C.

This module does not import streamlit, so it can be unit-tested directly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pandas as pd

from src.inference import get_model_metrics, get_required_feature_columns, predict_with_clipping

# -----------------------------------------------------------------------------
# Model-supported inputs (MODEL side, do not loosen)
# -----------------------------------------------------------------------------
# The six biomimetic tubercle geometries in data/processed/openfoam_phase2_cleaned.csv
# (all shape "whale"). Amplitude and wavelength are paired in the data, so they
# are offered as pairs, ordered by amplitude ("bump height").
TUBERCLE_GEOMETRIES: List[Dict[str, float]] = [
    {"tubercle_amplitude": 26.247, "tubercle_wavelength": 49.607},
    {"tubercle_amplitude": 28.537, "tubercle_wavelength": 47.334},
    {"tubercle_amplitude": 29.663, "tubercle_wavelength": 46.344},
    {"tubercle_amplitude": 30.421, "tubercle_wavelength": 45.873},
    {"tubercle_amplitude": 31.564, "tubercle_wavelength": 43.639},
    {"tubercle_amplitude": 32.734, "tubercle_wavelength": 42.337},
]
DEFAULT_BUMP_INDEX = 5            # tallest bumps; cannot reach the mainland, so Bump Height matters
WING_FAMILIES = ("symmetric", "cambered", "biomimetic")
ANGLES = list(range(0, 25))       # training grid: 0-24 deg in 1 deg steps
SPEEDS = (15, 30)                 # training grid: the only two airspeeds
FIXED_GEOMETRY = {"root_chord": 1.0, "tip_chord": 1.0, "sweep_angle": 0.0}
FALLBACK_RMSE = 0.0203

# -----------------------------------------------------------------------------
# Game rules (GAME side, invented, tune freely)
# -----------------------------------------------------------------------------
MAINLAND_DISTANCE = 1000          # metres from the island to the mainland coast
WIN_X_OVER_C = 0.90               # minimum predicted x/c that can ever reach the mainland
GLIDE_STEEPNESS = 6.0             # how strongly x/c changes glide efficiency
BEST_LAUNCH_ANGLE = 8.0           # launch power peaks at this angle
LAUNCH_ANGLE_SPREAD = 9.0         # how quickly power falls away from the best angle
SPEED_POWER = {15: 0.88, 30: 1.0}

STARTER_SLOTS: List[Dict[str, Any]] = [  # picked by the game; no player control at first
    {"name": "Driftwood Plank", "airfoil_family": "symmetric", "angle_of_attack": 4},
    {"name": "Palm Glider", "airfoil_family": "symmetric", "angle_of_attack": 8},
    {"name": "Sail Scrap", "airfoil_family": "cambered", "angle_of_attack": 8},
]
DEFAULT_SPEED = 15

# (key, points needed, title, description); unlocked automatically, in order.
UPGRADES: List[Dict[str, Any]] = [
    {"key": "wing_types", "points": 50, "title": "🪵 Wing Workshop",
     "text": "Choose symmetric or cambered for each of your three wings."},
    {"key": "angle", "points": 130, "title": "📐 Launch Ramp",
     "text": "Set the launch angle (angle of attack) for this round's wings."},
    {"key": "speed", "points": 220, "title": "💨 Turbo Catapult",
     "text": "Launch at 30 m/s for more launch power. Watch what it does to x/c!"},
    {"key": "biomimetic", "points": 320, "title": "🐋 Whale Wing",
     "text": "Unlock the biomimetic wing with humpback-whale tubercles."},
    {"key": "bump_height", "points": 480, "title": "⛰️ Bump Height",
     "text": "Pick from the six simulated tubercle designs (bump height and spacing)."},
    {"key": "wind_scout", "points": 700, "title": "🔭 Wind Scout",
     "text": "Preview the model's x/c for every wing before you launch."},
]
NEW_BEST_BONUS = 20


def unlocked(points: int) -> set:
    return {u["key"] for u in UPGRADES if points >= u["points"]}


def next_upgrade(points: int) -> Dict[str, Any] | None:
    return next((u for u in UPGRADES if points < u["points"]), None)


def launch_power(angle_of_attack: float, airspeed: int) -> float:
    """GAME RULE. Power from launch angle and speed, in (0, 1]."""
    angle_factor = 0.55 + 0.45 * math.exp(-((angle_of_attack - BEST_LAUNCH_ANGLE) / LAUNCH_ANGLE_SPREAD) ** 2)
    return min(1.0, angle_factor * SPEED_POWER[airspeed])


def glide_efficiency(x_over_c: float) -> float:
    """GAME RULE. Rises with x/c and equals 1.0 exactly at WIN_X_OVER_C."""
    return math.exp(GLIDE_STEEPNESS * (x_over_c - WIN_X_OVER_C))


def flight_distance(x_over_c: float, angle_of_attack: float, airspeed: int) -> float:
    return MAINLAND_DISTANCE * launch_power(angle_of_attack, airspeed) * glide_efficiency(x_over_c)


def flight_height(x_over_c: float) -> float:
    """GAME RULE. Peak height of the flight path in metres. Higher x/c flies higher."""
    return 20 + 180 * max(0.0, x_over_c) ** 2


# -----------------------------------------------------------------------------
# Building model inputs
# -----------------------------------------------------------------------------
def build_payload(airfoil_family: str, angle_of_attack: int, airspeed: int,
                  bump_index: int = DEFAULT_BUMP_INDEX) -> Dict[str, Any]:
    """Return the exact model input dict. Rejects anything outside the training grid."""
    if airfoil_family not in WING_FAMILIES:
        raise ValueError(f"Unsupported wing family: {airfoil_family}")
    if angle_of_attack not in ANGLES:
        raise ValueError(f"Unsupported angle of attack: {angle_of_attack}")
    if airspeed not in SPEEDS:
        raise ValueError(f"Unsupported airspeed: {airspeed}")
    if airfoil_family == "biomimetic":
        if not 0 <= bump_index < len(TUBERCLE_GEOMETRIES):
            raise ValueError(f"Unsupported tubercle design: {bump_index}")
        tubercles = {"tubercle_shape": "whale", **TUBERCLE_GEOMETRIES[bump_index]}
    else:
        tubercles = {"tubercle_shape": "none", "tubercle_amplitude": 0.0, "tubercle_wavelength": 0.0}
    payload = {"airfoil_family": airfoil_family, "angle_of_attack": float(angle_of_attack),
               "airspeed": airspeed, **tubercles, **FIXED_GEOMETRY}
    missing = set(get_required_feature_columns()) - set(payload)
    if missing:
        raise ValueError(f"Payload is missing model features: {sorted(missing)}")
    return payload


def check_allowed(slot: Dict[str, Any], airspeed: int, unlocks: set) -> None:
    """Make sure a slot only uses settings the player has unlocked."""
    if slot["airfoil_family"] == "biomimetic" and "biomimetic" not in unlocks:
        raise ValueError("The Whale Wing is not unlocked yet.")
    if slot.get("bump_index", DEFAULT_BUMP_INDEX) != DEFAULT_BUMP_INDEX and "bump_height" not in unlocks:
        raise ValueError("Bump Height is not unlocked yet.")
    if airspeed != DEFAULT_SPEED and "speed" not in unlocks:
        raise ValueError("The Turbo Catapult is not unlocked yet.")


def predict_x(slot: Dict[str, Any], airspeed: int) -> Dict[str, Any]:
    payload = build_payload(slot["airfoil_family"], int(slot["angle_of_attack"]), airspeed,
                            slot.get("bump_index", DEFAULT_BUMP_INDEX))
    raw, clipped, was_clipped = predict_with_clipping(payload)
    return {"inputs": payload, "raw": float(raw), "x_over_c": float(clipped), "was_clipped": bool(was_clipped)}


def fly(slots: List[Dict[str, Any]], airspeed: int, unlocks: set) -> List[Dict[str, Any]]:
    """Launch every slot. Returns one flight record per wing."""
    flights = []
    for slot in slots:
        check_allowed(slot, airspeed, unlocks)
        pred = predict_x(slot, airspeed)
        x = pred["x_over_c"]
        distance = flight_distance(x, slot["angle_of_attack"], airspeed)
        flights.append({
            "name": slot["name"], **pred,
            "launch_power": launch_power(slot["angle_of_attack"], airspeed),
            "glide_efficiency": glide_efficiency(x),
            "distance": min(distance, MAINLAND_DISTANCE + 150),   # land on the beach, not inland
            "height": flight_height(x),
            "reached_mainland": distance >= MAINLAND_DISTANCE,
        })
    return flights


def round_points(flights: List[Dict[str, Any]], previous_best: float) -> Dict[str, Any]:
    """GAME RULE. 1 point per 10 m flown by all three wings, +bonus for a new best."""
    base = int(round(sum(min(f["distance"], MAINLAND_DISTANCE) for f in flights) / 10))
    best = max(f["distance"] for f in flights)
    bonus = NEW_BEST_BONUS if best > previous_best else 0
    return {"base": base, "bonus": bonus, "total": base + bonus, "best": max(best, previous_best)}


def comparison_threshold() -> float:
    try:
        return float(get_model_metrics().get("rmse", FALLBACK_RMSE))
    except Exception:
        return FALLBACK_RMSE


def results_dataframe(rounds: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rnd in rounds:
        for f in rnd["flights"]:
            rows.append({
                "round": rnd["round"], "wing": f["name"], **f["inputs"],
                "model_separation_x_over_c": round(f["x_over_c"], 4),
                "model_raw_output": round(f["raw"], 4), "clipped": f["was_clipped"],
                "game_launch_power": round(f["launch_power"], 3),
                "game_glide_efficiency": round(f["glide_efficiency"], 3),
                "game_distance_m": round(f["distance"]), "reached_mainland": f["reached_mainland"],
                "round_points": rnd["points"]["total"],
            })
    df = pd.DataFrame(rows)
    df["note"] = ("x/c is an ML screening prediction (not CFD). Distance, power and points are "
                  "invented game rules, not physics.")
    return df
