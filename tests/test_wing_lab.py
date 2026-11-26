"""Tests for the Wing Lab game and the multipage router.

Run from the repo root:  python -m pytest tests -q
(pytest is a dev-only dependency and is not needed for deployment.)
"""

import itertools
import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app")]

from src.inference import predict_with_clipping  # noqa: E402
import wing_lab_core as game  # noqa: E402

ENTRY = str(ROOT / "app" / "app.py")
DATA = pd.read_csv(ROOT / "data/processed/autodesk_cfd_phase2_cleaned.csv")


@pytest.fixture(scope="module")
def predictions():
    """Model x/c for every input the game can ever send to the model."""
    out = {}
    for fam in game.WING_FAMILIES:
        bumps = range(len(game.TUBERCLE_GEOMETRIES)) if fam == "biomimetic" else [game.DEFAULT_BUMP_INDEX]
        for bump, angle, speed in itertools.product(bumps, game.ANGLES, game.SPEEDS):
            out[(fam, bump, angle, speed)] = predict_with_clipping(game.build_payload(fam, angle, speed, bump))[1]
    return out


def best_distance(predictions, unlocks):
    """Best single-wing distance a player can reach with a given set of unlocks."""
    best = 0.0
    for (fam, bump, angle, speed), x in predictions.items():
        if fam == "biomimetic" and "biomimetic" not in unlocks:
            continue
        if fam == "biomimetic" and bump != game.DEFAULT_BUMP_INDEX and "bump_height" not in unlocks:
            continue
        allowed_angles = game.ANGLES if "angle" in unlocks else [s["angle_of_attack"] for s in game.STARTER_SLOTS]
        if angle not in allowed_angles or (speed != game.DEFAULT_SPEED and "speed" not in unlocks):
            continue
        if "wing_types" not in unlocks and (fam, angle) not in {(s["airfoil_family"], s["angle_of_attack"])
                                                               for s in game.STARTER_SLOTS}:
            continue
        best = max(best, game.flight_distance(x, angle, speed))
    return best


# --- MODEL side: only supported inputs, backend unchanged -------------------------
def test_geometries_match_training_data_exactly():
    trained = set(DATA[DATA.airfoil_family == "biomimetic"][["tubercle_amplitude", "tubercle_wavelength"]]
                  .itertuples(index=False, name=None))
    offered = {(g["tubercle_amplitude"], g["tubercle_wavelength"]) for g in game.TUBERCLE_GEOMETRIES}
    assert offered == trained


def test_angles_speeds_families_match_training_grid():
    assert set(game.ANGLES) == set(DATA.angle_of_attack.unique())
    assert set(game.SPEEDS) == set(DATA.airspeed.unique())
    assert set(game.WING_FAMILIES) == set(DATA.airfoil_family.unique())


@pytest.mark.parametrize("args", [("delta", 8, 15), ("symmetric", 25, 15), ("symmetric", 8.5, 15),
                                  ("symmetric", 8, 20), ("biomimetic", 8, 15, 6)])
def test_unsupported_inputs_rejected(args):
    with pytest.raises(ValueError):
        game.build_payload(*args)


@pytest.mark.parametrize("config", [{"families": ["delta"] * 3}, {"angle": 30}, {"speed": 20},
                                    {"bumps": [9, 0, 0]}, {"families": ["symmetric"]}])
def test_bad_frontend_config_reports_error_without_crashing(config):
    state = game.new_state()
    game.handle_event(state, {"action": "launch", "config": config})
    assert state["error"] and state["rounds"] == []


def test_predict_x_uses_backend():
    slot = {"airfoil_family": "biomimetic", "angle_of_attack": 5, "bump_index": 0}
    _, clipped, _ = predict_with_clipping(game.build_payload("biomimetic", 5, 30, 0))
    assert game.predict_x(slot, 30)["x_over_c"] == pytest.approx(clipped, abs=1e-12)


def test_default_bump_is_the_worst_design(predictions):
    def mean_x(b):
        xs = [x for (f, bb, _, _), x in predictions.items() if f == "biomimetic" and bb == b]
        return sum(xs) / len(xs)
    assert game.DEFAULT_BUMP_INDEX == min(range(len(game.TUBERCLE_GEOMETRIES)), key=mean_x)


# --- GAME side: the rules promised to the user --------------------------------------
def test_distance_is_proportional_to_x_over_c():
    for angle, speed in itertools.product(game.ANGLES, game.SPEEDS):
        d1, d2 = game.flight_distance(0.6, angle, speed), game.flight_distance(0.9, angle, speed)
        assert d2 / d1 == pytest.approx(1.5)
    heights = [game.flight_height(x / 100) for x in range(0, 101)]
    assert all(b > a for a, b in zip(heights, heights[1:]))


def test_launch_power_capped_so_power_alone_cannot_win():
    assert max(game.launch_power(a, s) for a in game.ANGLES for s in game.SPEEDS) <= 1.0
    for x in [i / 1000 for i in range(0, int(game.WIN_X_OVER_C * 1000))]:
        assert max(game.flight_distance(x, a, s) for a in game.ANGLES for s in game.SPEEDS) < game.MAINLAND_DISTANCE


def test_only_high_x_over_c_wins_and_game_is_winnable(predictions):
    winners = [k for k, x in predictions.items() if game.flight_distance(x, k[2], k[3]) >= game.MAINLAND_DISTANCE]
    assert winners
    assert all(predictions[k] >= game.WIN_X_OVER_C and k[0] == "biomimetic" for k in winners)
    assert all(k[1] != game.DEFAULT_BUMP_INDEX for k in winners), "Bump Height must be needed to win"


def test_no_soft_lock_every_upgrade_is_reachable(predictions):
    """With the upgrades unlocked so far, some wing always earns enough for the next one."""
    for i, upgrade in enumerate(game.UPGRADES):
        have = {u["key"] for u in game.UPGRADES[:i]}
        reachable_points = round(best_distance(predictions, have) / game.METRES_PER_POINT)
        assert reachable_points >= upgrade["points"], upgrade["key"]
    assert best_distance(predictions, {u["key"] for u in game.UPGRADES}) >= game.MAINLAND_DISTANCE


def test_points_only_for_new_records():
    state = game.new_state()
    game.handle_event(state, {"action": "launch", "config": None})
    first = state["points"]
    assert first > 0 and state["rounds"][-1]["new_record"]
    for _ in range(3):                                   # same wings again: no farming
        game.handle_event(state, {"action": "launch", "config": None})
        assert state["rounds"][-1]["points_earned"] == 0
    assert state["points"] == first == round(state["best"] / game.METRES_PER_POINT)


def test_locked_settings_are_forced_back_to_defaults():
    state = game.new_state()
    cheat = {"families": ["biomimetic"] * 3, "bumps": [0, 0, 0], "angle": 2, "speed": 30}
    game.handle_event(state, {"action": "launch", "config": cheat})
    assert state["rounds"][-1]["config"] == game.default_config()
    game.handle_event(state, {"action": "scout", "config": cheat})
    assert state["error"] and state["scout"] is None


def test_full_game_through_events_and_rescue():
    state = game.new_state()
    game.handle_event(state, {"action": "intro_done"})
    plan = [None,
            {"families": ["cambered"] * 3},
            {"families": ["cambered"] * 3, "angle": 1},
            {"families": ["cambered"] * 3, "angle": 1, "speed": 30},
            {"families": ["biomimetic"] * 3, "angle": 2, "speed": 30, "bumps": [5, 5, 5]},
            {"families": ["biomimetic"] * 3, "angle": 2, "speed": 30, "bumps": [0, 0, 0]}]
    for cfg in plan:
        full = {**game.default_config(), **(cfg or {})}
        game.handle_event(state, {"action": "launch", "config": full})
        assert state["error"] is None
    assert state["won"] and not state["rescue_seen"]
    assert game.unlocked(state["points"]) == {u["key"] for u in game.UPGRADES}
    rounds_before = len(state["rounds"])
    game.handle_event(state, {"action": "launch", "config": None})   # no launching after the win
    assert len(state["rounds"]) == rounds_before
    game.handle_event(state, {"action": "rescue_done"})
    assert state["rescue_seen"]


def _play(configs):
    state = game.new_state()
    for cfg in configs:
        game.handle_event(state, {"action": "launch", "config": {**game.default_config(), **(cfg or {})}})
    return state


def test_every_round_has_a_lesson_without_performance_claims():
    state = _play([None, {"families": ["cambered"] * 3}, {"families": ["cambered"] * 3},
                   {"families": ["cambered", "symmetric", "cambered"], "angle": 1},
                   {"families": ["cambered"] * 3, "angle": 1, "speed": 30},
                   {"families": ["biomimetic", "cambered", "biomimetic"], "angle": 2, "speed": 30},
                   {"families": ["biomimetic"] * 3, "angle": 2, "speed": 30, "bumps": [0, 3, 5]}])
    banned = ["lift", "drag", "efficien", "optimal", "better wing", "best wing"]
    for rnd in state["rounds"]:
        les = rnd["lesson"]
        assert les["title"] and les["explain"] and les["observation"]
        text = " ".join([les["observation"], les["explain"], les["fair"] or "", les["tip"] or ""]).lower()
        assert not any(b in text for b in banned), (rnd["round"], text)
        assert les["sketch"]["x"] == round(max(f["x_over_c"] for f in rnd["flights"]), 3)


def test_lesson_finds_fair_tests():
    state = _play([None, {"families": ["cambered"] * 3}, {"families": ["cambered"] * 3},
                   {"families": ["cambered", "symmetric", "cambered"], "angle": 1}])
    les = state["rounds"][-1]["lesson"]
    assert les["concept"] == "camber" and "only the wing shape differed" in les["fair"]
    state = _play([None, None])
    assert state["rounds"][-1]["lesson"]["concept"] in ("angle", "repeat", "camber")


def test_lesson_flags_changing_many_things():
    state = _play([None, {"families": ["cambered"] * 3}, {"families": ["cambered"] * 3},
                   {"families": ["cambered"] * 3, "angle": 1},
                   {"families": ["symmetric"] * 3, "angle": 12, "speed": 30}])
    les = state["rounds"][-1]["lesson"]
    assert les["tip"] and "one thing at a time" in les["tip"]


def test_wing_names_have_no_materials():
    assert all("name" not in s for s in game.STARTER_SLOTS)


def test_results_dataframe_labels_game_values():
    state = game.new_state()
    game.handle_event(state, {"action": "launch", "config": None})
    df = game.results_dataframe(state["rounds"])
    assert len(df) == 3 and df["note"].str.contains("invented game rules").all()


# --- app-level flows ---------------------------------------------------------------
def _app():
    at = AppTest.from_file(ENTRY, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _send(at, action, nonce, config=None):
    at.session_state["wl_game_event"] = {"action": action, "nonce": nonce, "config": config}
    at.run()
    assert not at.exception, at.exception


def test_home_is_default_page():
    at = _app()
    assert at.title[0].value == "🛩️ WingCheck"


def test_wing_lab_page_processes_events_once_and_restarts():
    at = _app()
    at.switch_page("wing_lab.py").run()
    assert not at.exception
    _send(at, "launch", "n1")
    assert len(at.session_state["wl_state"]["rounds"]) == 1
    at.run()                                          # same nonce again: not re-processed
    assert len(at.session_state["wl_state"]["rounds"]) == 1
    assert any("Total: 61" in c.value for c in at.caption)
    gen = at.session_state["wl_gen"]
    _send(at, "restart", "n2")
    assert at.session_state["wl_state"]["rounds"] == [] and at.session_state["wl_gen"] == gen + 1


def test_debrief_and_quiz_after_rescue():
    at = _app()
    at.switch_page("wing_lab.py").run()
    state = game.new_state()
    for cfg in [None, {"families": ["cambered"] * 3}, {"families": ["cambered"] * 3, "angle": 1},
                {"families": ["cambered"] * 3, "angle": 1, "speed": 30},
                {"families": ["biomimetic"] * 3, "angle": 2, "speed": 30},
                {"families": ["biomimetic"] * 3, "angle": 2, "speed": 30, "bumps": [0, 0, 0]}]:
        game.handle_event(state, {"action": "launch", "config": {**game.default_config(), **(cfg or {})}})
    at.session_state["wl_state"] = state
    at.run()
    assert not any("Debrief" in h.value for h in at.header)      # not before the rescue animation
    _send(at, "rescue_done", "r1")
    assert any("Debrief" in h.value for h in at.header)
    at.radio(key="wl_quiz_choice").set_value(1).run()
    next(b for b in at.button if b.label == "Check my answer").click().run()
    assert any(s.value.startswith("Correct") for s in at.success)


def test_advanced_mode_still_runs_and_predicts():
    at = _app()
    at.switch_page("advanced_mode.py").run()
    assert not at.exception
    next(b for b in at.button if b.label == "Launch WingCheck").click().run()
    next(b for b in at.button if b.label == "Start Here").click().run()
    at.button(key="FormSubmitter:input_form-Run Prediction").click().run()
    assert not at.exception
    assert at.session_state["latest_prediction"] is not None
