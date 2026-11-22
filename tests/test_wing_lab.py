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
DATA = pd.read_csv(ROOT / "data/processed/openfoam_phase2_cleaned.csv")


def all_supported_configs():
    for fam in game.WING_FAMILIES:
        bumps = range(len(game.TUBERCLE_GEOMETRIES)) if fam == "biomimetic" else [game.DEFAULT_BUMP_INDEX]
        for bump, angle, speed in itertools.product(bumps, game.ANGLES, game.SPEEDS):
            yield fam, bump, angle, speed


@pytest.fixture(scope="module")
def predictions():
    return {cfg: predict_with_clipping(game.build_payload(cfg[0], cfg[2], cfg[3], cfg[1]))[1]
            for cfg in all_supported_configs()}


# --- MODEL side: only supported inputs, backend unchanged -------------------------
def test_geometries_match_training_data_exactly():
    trained = set(DATA[DATA.airfoil_family == "biomimetic"][["tubercle_amplitude", "tubercle_wavelength"]]
                  .itertuples(index=False, name=None))
    offered = {(g["tubercle_amplitude"], g["tubercle_wavelength"]) for g in game.TUBERCLE_GEOMETRIES}
    assert offered == trained


def test_angles_and_speeds_match_training_grid():
    assert set(game.ANGLES) == set(DATA.angle_of_attack.unique())
    assert set(game.SPEEDS) == set(DATA.airspeed.unique())
    assert set(game.WING_FAMILIES) == set(DATA.airfoil_family.unique())


@pytest.mark.parametrize("args", [("delta", 8, 15), ("symmetric", 25, 15), ("symmetric", 8.5, 15),
                                  ("symmetric", 8, 20), ("biomimetic", 8, 15, 6)])
def test_unsupported_inputs_rejected(args):
    with pytest.raises(ValueError):
        game.build_payload(*args)


def test_predict_x_uses_backend():
    slot = {"airfoil_family": "biomimetic", "angle_of_attack": 5, "bump_index": 0}
    raw, clipped, _ = predict_with_clipping(game.build_payload("biomimetic", 5, 30, 0))
    assert game.predict_x(slot, 30)["x_over_c"] == pytest.approx(clipped, abs=1e-12)


def test_locked_settings_rejected():
    bio = {"name": "x", "airfoil_family": "biomimetic", "angle_of_attack": 4}
    with pytest.raises(ValueError):
        game.fly([bio], 15, unlocks=set())
    sym = {"name": "x", "airfoil_family": "symmetric", "angle_of_attack": 4}
    with pytest.raises(ValueError):
        game.fly([sym], 30, unlocks=set())


# --- GAME side: the rules promised to the user --------------------------------------
def test_distance_increases_with_x_over_c():
    for angle, speed in itertools.product(game.ANGLES, game.SPEEDS):
        d = [game.flight_distance(x / 100, angle, speed) for x in range(0, 101)]
        assert all(b > a for a, b in zip(d, d[1:]))
    h = [game.flight_height(x / 100) for x in range(0, 101)]
    assert all(b > a for a, b in zip(h, h[1:]))


def test_launch_power_capped_so_power_alone_cannot_win():
    assert max(game.launch_power(a, s) for a in game.ANGLES for s in game.SPEEDS) <= 1.0
    for x in [i / 1000 for i in range(0, int(game.WIN_X_OVER_C * 1000))]:
        assert max(game.flight_distance(x, a, s) for a in game.ANGLES for s in game.SPEEDS) < game.MAINLAND_DISTANCE


def test_only_high_x_over_c_wins_and_game_is_winnable(predictions):
    winners = [cfg for cfg, x in predictions.items()
               if game.flight_distance(x, cfg[2], cfg[3]) >= game.MAINLAND_DISTANCE]
    assert winners, "the game must be winnable"
    assert all(predictions[cfg] >= game.WIN_X_OVER_C for cfg in winners)
    assert all(cfg[0] == "biomimetic" for cfg in winners)


def test_bump_height_upgrade_is_needed_to_win(predictions):
    default_wins = [cfg for cfg, x in predictions.items()
                    if cfg[0] == "biomimetic" and cfg[1] == game.DEFAULT_BUMP_INDEX
                    and game.flight_distance(x, cfg[2], cfg[3]) >= game.MAINLAND_DISTANCE]
    assert default_wins == []


def test_first_round_unlocks_first_upgrade():
    flights = game.fly(game.STARTER_SLOTS, game.DEFAULT_SPEED, set())
    assert game.round_points(flights, 0)["total"] >= game.UPGRADES[0]["points"]


def test_unlock_order():
    assert [u["key"] for u in game.UPGRADES] == ["wing_types", "angle", "speed", "biomimetic",
                                                 "bump_height", "wind_scout"]
    points = [u["points"] for u in game.UPGRADES]
    assert points == sorted(points)


def test_results_dataframe_labels_game_values():
    flights = game.fly(game.STARTER_SLOTS, 15, set())
    rounds = [{"round": 1, "flights": flights, "points": game.round_points(flights, 0)}]
    df = game.results_dataframe(rounds)
    assert len(df) == 3 and df["note"].str.contains("invented game rules").all()


# --- app-level flows ---------------------------------------------------------------
def _app():
    at = AppTest.from_file(ENTRY, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _launch(at):
    next(b for b in at.button if b.label == "🚀 Launch!").click().run()
    assert not at.exception, at.exception


def test_home_is_default_page():
    at = _app()
    assert at.title[0].value == "🛩️ WingCheck"


def test_full_game_can_be_won():
    at = _app()
    at.switch_page("wing_lab.py").run()
    assert not at.exception
    assert not at.selectbox and not at.slider   # nothing to control at the start

    _launch(at)
    assert at.session_state["wl_points"] >= 50
    assert len(at.selectbox) == 3                # Wing Workshop unlocked

    # Keep flying the best available wings until every upgrade is unlocked.
    for _ in range(30):
        if at.session_state["wl_won"]:
            break
        unl = game.unlocked(at.session_state["wl_points"])
        fam = "biomimetic" if "biomimetic" in unl else "cambered"
        for i in range(3):
            at.selectbox(key=f"wl_family_{i}").set_value(fam)
        at.run()
        if "angle" in unl:
            at.slider(key="wl_angle").set_value(3 if fam == "biomimetic" else 1)
        if "speed" in unl:
            at.radio(key="wl_speed").set_value(30)
        if "bump_height" in unl:
            for i in range(3):
                at.select_slider(key=f"wl_bump_{i}").set_value(0)
        at.run()
        _launch(at)
    assert at.session_state["wl_won"]
    assert any(h.value == "🎉 Kai is rescued!" for h in at.header)

    at.radio(key="wl_quiz_choice").set_value(1).run()
    next(b for b in at.button if b.label == "Check my answer").click().run()
    assert at.success and any(s.value.startswith("Correct") for s in at.success)

    next(b for b in at.button if b.label.startswith("🔄")).click().run()
    assert at.session_state["wl_points"] == 0 and at.session_state["wl_rounds"] == []


def test_advanced_mode_still_runs_and_predicts():
    at = _app()
    at.switch_page("advanced_mode.py").run()
    assert not at.exception
    next(b for b in at.button if b.label == "Launch WingCheck").click().run()
    next(b for b in at.button if b.label == "Start Here").click().run()
    at.button(key="FormSubmitter:input_form-Run Prediction").click().run()
    assert not at.exception
    assert at.session_state["latest_prediction"] is not None
