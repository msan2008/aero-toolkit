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
# The six biomimetic tubercle geometries in data/processed/autodesk_cfd_phase2_cleaned.csv
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
    {"airfoil_family": "symmetric", "angle_of_attack": 4},
    {"airfoil_family": "symmetric", "angle_of_attack": 8},
    {"airfoil_family": "cambered", "angle_of_attack": 8},
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
        state["rounds"][-1]["lesson"] = round_lesson(state["rounds"])
        if any(f["reached_mainland"] for f in flights):
            state["won"] = True
    return state


# -----------------------------------------------------------------------------
# Lab notes: one short aerodynamics lesson after every round
# -----------------------------------------------------------------------------
# The lesson is chosen from what actually happened: it looks for wings that
# differ in exactly ONE setting (a fair test) and explains that setting. Every
# observation quotes the model's own x/c values; explanations describe general
# aerodynamics and never claim that a wing has more lift, less drag, or is better.
CONCEPTS: Dict[str, Dict[str, str]] = {
    "separation": {"title": "Flow separation", "text": (
        "Air flowing over a wing tries to follow its curved top surface. Where it can't keep up, it breaks "
        "away into swirling, messy air. That spot is the separation point, x/c: 0 is the front edge, 1 is "
        "the back edge. The farther back it is, the more of the wing still has smooth air flowing over it.")},
    "camber": {"title": "Wing shape (camber)", "text": (
        "A symmetric wing is the same on top and bottom. A cambered wing is arched on top, like a bird's "
        "wing. The shape changes how the air pressure varies along the surface, and that decides how far "
        "the air can follow the wing before it breaks away.")},
    "angle": {"title": "Angle of attack", "text": (
        "Tilting a wing more steeply makes the air work harder to stay on the top surface as it flows toward "
        "the back. The steeper the tilt, the sooner it gives up and separates. Tilt too far and most of the "
        "top is covered in separated air: that is called a stall.")},
    "speed": {"title": "Airspeed", "text": (
        "Speed changes how the thin layer of air touching the wing behaves, in ways that aren't always "
        "obvious (scientists use a number called the Reynolds number to describe it). In these CFD results "
        "faster was not automatically better, which is why engineers test instead of guessing.")},
    "tubercles": {"title": "Tubercles (whale bumps)", "text": (
        "The bumps on a humpback whale's flipper stir up small swirls of air called vortices. These pull "
        "faster-moving air down close to the surface and give the slow air there extra energy, which can "
        "help the flow stay attached longer.")},
    "bumps": {"title": "Bump size", "text": (
        "Bigger bumps are not automatically better. The height and spacing of tubercles change the swirls "
        "they make, so engineers test many sizes to find what works for each situation.")},
    "repeat": {"title": "Repeatable results", "text": (
        "Same wing, same settings, same answer: the model gives the same prediction every time. To learn "
        "something new, change one thing and compare.")},
}
_VAR_NAME = {"family": "wing shape", "bump": "bump size", "angle": "launch angle", "speed": "launch speed"}
_FAMILY_WORD = {"symmetric": "symmetric", "cambered": "cambered", "biomimetic": "whale"}


def _settings(f: Dict[str, Any]) -> Dict[str, Any]:
    i = f["inputs"]
    return {"family": i["airfoil_family"], "angle": int(i["angle_of_attack"]), "speed": i["airspeed"],
            "bump": round(i["tubercle_amplitude"], 1) if i["airfoil_family"] == "biomimetic" else None}


def _diff(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    keys = ("family", "angle", "speed") if a["family"] != b["family"] else ("bump", "angle", "speed")
    return [k for k in keys if a[k] != b[k]]                      # bump size only compares two whale wings


def _observe(var: str, hi: Dict[str, Any], lo: Dict[str, Any], threshold: float) -> Dict[str, str]:
    """Describe a fair comparison between two flights that differ only in `var`."""
    sh, sl, xh, xl = _settings(hi), _settings(lo), hi["x_over_c"], lo["x_over_c"]
    label = {"family": lambda s: f"on the {_FAMILY_WORD[s['family']]} wing", "angle": lambda s: f"at {s['angle']}°",
             "speed": lambda s: f"at {s['speed']} m/s", "bump": lambda s: f"with {s['bump']} mm bumps"}[var]
    if abs(xh - xl) <= threshold:
        obs = (f"Changing only the {_VAR_NAME[var]} barely mattered here: {label(sh)} the model predicts x/c "
               f"{xh:.2f}, {label(sl)} x/c {xl:.2f}. That's within the model's typical error.")
    else:
        obs = (f"{label(sh).capitalize()}, the air stayed attached to x/c {xh:.2f}; {label(sl)}, it separated "
               f"sooner, at x/c {xl:.2f}.")
        if var == "angle" and sh["angle"] > sl["angle"]:
            obs += " Surprise: here the steeper wing kept the air attached longer. Real airflow doesn't always follow simple rules."
        if var == "bump" and sh["bump"] > sl["bump"]:
            obs += " Here the taller bumps helped more."
    if var == "family":
        concept = "tubercles" if "biomimetic" in (sh["family"], sl["family"]) else "camber"
        obs = obs.replace(", the air", " the air", 1) if obs.startswith("On the") else obs
    else:
        concept = {"angle": "angle", "speed": "speed", "bump": "bumps"}[var]
    return {"concept": concept, "observation": obs}


def round_lesson(rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pick one lesson for the latest round, based on a fair comparison when there is one."""
    rnd, prev = rounds[-1], (rounds[-2] if len(rounds) > 1 else None)
    flights, threshold = rnd["flights"], comparison_threshold()
    best = max(flights, key=lambda f: f["x_over_c"])
    lesson: Dict[str, Any] = {"fair": None, "tip": None, "sketch": {
        "family": best["inputs"]["airfoil_family"], "angle": int(best["inputs"]["angle_of_attack"]),
        "x": round(best["x_over_c"], 3), "wing": best["name"]}}

    # 1) within this round: two wings that differ in exactly one setting
    priority = ["family", "bump", "angle", "speed"]
    pairs = []
    for i in range(len(flights)):
        for j in range(i + 1, len(flights)):
            d = _diff(_settings(flights[i]), _settings(flights[j]))
            if len(d) == 1:
                a, b = sorted((flights[i], flights[j]), key=lambda f: -f["x_over_c"])
                pairs.append((priority.index(d[0]), -abs(a["x_over_c"] - b["x_over_c"]), i, j, d[0], a, b))
    if rnd["round"] == 1:
        lesson.update(concept="separation", observation=(
            f"{best['name']} kept smooth air over the first {best['x_over_c'] * 100:.0f}% of its top surface. "
            f"{min(flights, key=lambda f: f['x_over_c'])['name']} lost it sooner, at "
            f"{min(f['x_over_c'] for f in flights) * 100:.0f}%. In this game, the farther back the air "
            "separates, the farther the glider glides."))
    elif pairs:
        _, _, _, _, var, hi, lo = min(pairs, key=lambda p: p[:4])
        lesson.update(_observe(var, hi, lo, threshold))
        lesson["fair"] = f"Fair test: {hi['name']} and {lo['name']} differed only in {_VAR_NAME[var]}."
    elif prev is not None:
        # 2) all three wings alike: compare with the previous round
        now_s = [_settings(f) for f in flights]
        prev_s = [_settings(f) for f in prev["flights"]]
        changes = sorted({k for a, b in zip(now_s, prev_s) for k in _diff(a, b)}, key=priority.index)
        nb, pb = best, max(prev["flights"], key=lambda f: f["x_over_c"])
        if not changes:
            lesson.update(concept="repeat", observation=(
                f"You launched the same wings as last round, and the model predicted the same x/c "
                f"({nb['x_over_c']:.2f}) again."))
        elif len(changes) == 1:
            hi, lo = (nb, pb) if nb["x_over_c"] >= pb["x_over_c"] else (pb, nb)
            lesson.update(_observe(changes[0], hi, lo, threshold))
            lesson["observation"] = "Compared with last round: " + lesson["observation"][0].lower() + lesson["observation"][1:]
            lesson["fair"] = f"Fair test: since last round you changed only the {_VAR_NAME[changes[0]]}."
        else:
            lesson.update(concept="camber" if nb["inputs"]["airfoil_family"] != "biomimetic" else "tubercles",
                          observation=f"This round's latest separation point was x/c {nb['x_over_c']:.2f} "
                                      f"(last round: {pb['x_over_c']:.2f}).")
            lesson["tip"] = (f"You changed {len(changes)} things at once ({', '.join(_VAR_NAME[c] for c in changes)}), "
                             "so it's hard to tell which one made the difference. Try changing one thing at a time.")
    else:
        lesson.update(concept="separation", observation=f"{best['name']} reached x/c {best['x_over_c']:.2f}.")
    c = CONCEPTS[lesson["concept"]]
    lesson.update(title=c["title"], explain=c["text"])
    return lesson


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
