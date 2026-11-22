"""Wing Lab game rules and game state, kept separate from the Streamlit UI.

Two kinds of numbers live in this game, and they must never be confused:

  * MODEL (real): separation_x_over_c, predicted by the unchanged
    src.inference.predict_with_clipping() from model-supported inputs only.
  * GAME (invented): flight distance, height, points and unlocks. These are
    made-up game rules, not physics, and are labelled as such in the UI.

Game rule (distance is proportional to x/c):
    distance = (MAINLAND_DISTANCE / WIN_X_OVER_C) * launch_power * x/c
  * launch_power (0-1] comes from launch angle and speed only, capped at 1.
  * Reaching the mainland needs launch_power * x/c >= WIN_X_OVER_C. Because
    launch_power <= 1, only wings with predicted x/c >= WIN_X_OVER_C can win.

Points rule: points are only awarded for beating your own record, 1 point
per 10 m of improvement, so repeating weak designs earns nothing. As a
result, total points always equal round(record / 10).

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
# The tallest bumps have the lowest average predicted x/c of the six, so the
# game starts there and students discover better designs themselves.
DEFAULT_BUMP_INDEX = 5
WING_FAMILIES = ("symmetric", "cambered", "biomimetic")
ANGLES = list(range(0, 25))       # training grid: 0-24 deg in 1 deg steps
SPEEDS = (15, 30)                 # training grid: the only two airspeeds
FIXED_GEOMETRY = {"root_chord": 1.0, "tip_chord": 1.0, "sweep_angle": 0.0}
FALLBACK_RMSE = 0.0203

# -----------------------------------------------------------------------------
# Game rules (GAME side, invented; tuned against the model so the game is winnable)
# -----------------------------------------------------------------------------
CHARACTER = "Link"
MAINLAND_DISTANCE = 1000          # metres from the island to the mainland coast
WIN_X_OVER_C = 0.90               # minimum predicted x/c that can ever reach the mainland
DISTANCE_SCALE = MAINLAND_DISTANCE / WIN_X_OVER_C
BEST_LAUNCH_ANGLE = 4.0           # launch power peaks at this angle
LAUNCH_ANGLE_SPREAD = 12.0        # how quickly power falls away from the best angle
MIN_ANGLE_POWER = 0.6
SPEED_POWER = {15: 0.9, 30: 1.0}
METRES_PER_POINT = 10

STARTER_SLOTS: List[Dict[str, Any]] = [  # picked by the game; no player control at first
    {"name": "Driftwood Plank", "airfoil_family": "symmetric", "angle_of_attack": 4},
    {"name": "Palm Glider", "airfoil_family": "symmetric", "angle_of_attack": 8},
    {"name": "Sail Scrap", "airfoil_family": "cambered", "angle_of_attack": 8},
]
DEFAULT_SPEED = 15
DEFAULT_ANGLE = 8

# Unlocked automatically when total points (= record / 10) reach the threshold.
# Each threshold is below the best flight possible with the upgrades before it
# (checked in tests), so a player can never get stuck.
UPGRADES: List[Dict[str, Any]] = [
    {"key": "wing_types", "points": 60, "title": "🪵 Wing Workshop",
     "text": "Choose symmetric or cambered for each of your three wings."},
    {"key": "angle", "points": 66, "title": "📐 Launch Ramp",
     "text": "Set the launch angle (angle of attack) for this round's wings."},
    {"key": "speed", "points": 71, "title": "💨 Turbo Catapult",
     "text": "Launch at 30 m/s for more launch power. Watch what it does to x/c!"},
    {"key": "biomimetic", "points": 75, "title": "🐋 Whale Wing",
     "text": "Unlock the biomimetic wing with humpback-whale tubercles."},
    {"key": "bump_height", "points": 95, "title": "⛰️ Bump Height",
     "text": "Pick from the six simulated tubercle designs (bump height and spacing)."},
    {"key": "wind_scout", "points": 97, "title": "🔭 Wind Scout",
     "text": "Preview the model's x/c for every wing before you launch."},
]


def unlocked(points: int) -> set:
    return {u["key"] for u in UPGRADES if points >= u["points"]}


def next_upgrade(points: int) -> Dict[str, Any] | None:
    return next((u for u in UPGRADES if points < u["points"]), None)


def launch_power(angle_of_attack: float, airspeed: int) -> float:
    """GAME RULE. Power from launch angle and speed, in (0, 1]."""
    bell = math.exp(-((angle_of_attack - BEST_LAUNCH_ANGLE) / LAUNCH_ANGLE_SPREAD) ** 2)
    return min(1.0, (MIN_ANGLE_POWER + (1 - MIN_ANGLE_POWER) * bell) * SPEED_POWER[airspeed])


def flight_distance(x_over_c: float, angle_of_attack: float, airspeed: int) -> float:
    """GAME RULE. Proportional to x/c: 1.5x the x/c flies 1.5x as far."""
    return DISTANCE_SCALE * launch_power(angle_of_attack, airspeed) * max(0.0, x_over_c)


def flight_height(x_over_c: float) -> float:
    """GAME RULE. Peak height of the flight path in metres. Higher x/c flies higher."""
    return 20 + 180 * max(0.0, x_over_c) ** 2


def points_for(best_after: float, best_before: float) -> int:
    """GAME RULE. Points only for beating your record: 1 point per 10 m improved."""
    if best_after <= best_before:
        return 0
    return round(best_after / METRES_PER_POINT) - round(best_before / METRES_PER_POINT)


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
        if bump_index not in range(len(TUBERCLE_GEOMETRIES)):
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


def default_config() -> Dict[str, Any]:
    return {"families": [s["airfoil_family"] for s in STARTER_SLOTS],
            "bumps": [DEFAULT_BUMP_INDEX] * len(STARTER_SLOTS),
            "angle": DEFAULT_ANGLE, "speed": DEFAULT_SPEED}


def sanitize_config(config: Dict[str, Any] | None, unlocks: set) -> Dict[str, Any]:
    """Turn a config sent by the game frontend into one the player is allowed to use.

    Locked settings are forced back to game defaults, so the browser can never
    unlock anything early. Values outside the training grid raise ValueError.
    """
    config = config or {}
    base = default_config()
    families = list(config.get("families", base["families"]))[:3]
    bumps = [int(b) for b in list(config.get("bumps", base["bumps"]))[:3]]
    angle, speed = int(config.get("angle", base["angle"])), int(config.get("speed", base["speed"]))
    if len(families) != 3 or len(bumps) != 3:
        raise ValueError("A launch needs exactly three wings.")
    for f, b in zip(families, bumps):
        if f not in WING_FAMILIES or b not in range(len(TUBERCLE_GEOMETRIES)):
            raise ValueError(f"Unsupported wing settings: {f}, bump {b}")
    if angle not in ANGLES or speed not in SPEEDS:
        raise ValueError(f"Unsupported launch settings: {angle} deg, {speed} m/s")

    if "wing_types" not in unlocks:
        families = list(base["families"])
    families = [f if (f != "biomimetic" or "biomimetic" in unlocks) else base["families"][i]
                for i, f in enumerate(families)]
    if "bump_height" not in unlocks:
        bumps = list(base["bumps"])
    if "angle" not in unlocks:
        angle = base["angle"]
    if "speed" not in unlocks:
        speed = base["speed"]
    return {"families": families, "bumps": bumps, "angle": angle, "speed": speed}


def slots_from_config(config: Dict[str, Any], unlocks: set) -> List[Dict[str, Any]]:
    """Three wing slots. Before the Launch Ramp, each starter keeps its own angle."""
    slots = []
    for i, starter in enumerate(STARTER_SLOTS):
        angle = config["angle"] if "angle" in unlocks else starter["angle_of_attack"]
        slots.append({"name": f"Wing {'ABC'[i]}", "airfoil_family": config["families"][i],
                      "angle_of_attack": int(angle), "bump_index": config["bumps"][i]})
    return slots


def predict_x(slot: Dict[str, Any], airspeed: int) -> Dict[str, Any]:
    payload = build_payload(slot["airfoil_family"], int(slot["angle_of_attack"]), airspeed,
                            slot.get("bump_index", DEFAULT_BUMP_INDEX))
    raw, clipped, was_clipped = predict_with_clipping(payload)
    return {"inputs": payload, "raw": float(raw), "x_over_c": float(clipped), "was_clipped": bool(was_clipped)}


def fly(slots: List[Dict[str, Any]], airspeed: int) -> List[Dict[str, Any]]:
    """Launch every slot. Returns one flight record per wing."""
    flights = []
    for slot in slots:
        pred = predict_x(slot, airspeed)
        x = pred["x_over_c"]
        distance = flight_distance(x, slot["angle_of_attack"], airspeed)
        flights.append({
            "name": slot["name"], **pred,
            "launch_power": launch_power(slot["angle_of_attack"], airspeed),
            "distance": min(distance, MAINLAND_DISTANCE + 150),   # land on the beach, not inland
            "height": flight_height(x),
            "reached_mainland": distance >= MAINLAND_DISTANCE,
        })
    return flights


# -----------------------------------------------------------------------------
# Game state machine (driven by events from the game frontend)
# -----------------------------------------------------------------------------
def new_state() -> Dict[str, Any]:
    return {"points": 0, "best": 0.0, "rounds": [], "won": False, "intro_seen": False,
            "rescue_seen": False, "config": default_config(), "scout": None, "error": None}


def handle_event(state: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Apply one frontend event ("intro_done", "launch", "scout", "rescue_done")."""
    action = event.get("action")
    state["error"] = None
    if action == "intro_done":
        state["intro_seen"] = True
    elif action == "rescue_done":
        state["rescue_seen"] = state["won"]
    elif action in ("launch", "scout"):
        if action == "launch" and state["won"]:
            return state
        unlocks = unlocked(state["points"])
        try:
            config = sanitize_config(event.get("config"), unlocks)
            slots = slots_from_config(config, unlocks)
            if action == "scout":
                if "wind_scout" not in unlocks:
                    raise ValueError("Wind Scout is not unlocked yet.")
                state["scout"] = {"config": config,
                                  "x": [round(predict_x(s, config["speed"])["x_over_c"], 3) for s in slots]}
                return state
            flights = fly(slots, config["speed"])
        except Exception as exc:  # bad input or model problem: report, don't crash
            state["error"] = str(exc)
            return state
        best_before = state["best"]
        best_after = max([best_before] + [f["distance"] for f in flights])
        earned = points_for(best_after, best_before)
        new_keys = unlocked(state["points"] + earned) - unlocks
        state["points"] += earned
        state["best"] = best_after
        state["config"] = config
        state["scout"] = None
        state["rounds"].append({
            "round": len(state["rounds"]) + 1, "flights": flights, "config": config,
            "points_earned": earned, "new_record": best_after > best_before,
            "total_points": state["points"],
            "new_unlocks": [u for u in UPGRADES if u["key"] in new_keys],
        })
        if any(f["reached_mainland"] for f in flights):
            state["won"] = True
    return state


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
                "game_distance_m": round(f["distance"]), "reached_mainland": f["reached_mainland"],
                "round_points_earned": rnd["points_earned"], "total_points": rnd["total_points"],
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["note"] = ("x/c is an ML screening prediction (not CFD). Distance, power and points are "
                      "invented game rules, not physics.")
    return df
