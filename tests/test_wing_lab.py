"""Tests for Wing Lab and the multipage router.

Run from the repo root:  python -m pytest tests -q
(pytest is a dev-only dependency and is not needed for deployment.)
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "app")]

from src.inference import predict_with_clipping  # noqa: E402
import wing_lab_core as core  # noqa: E402

ENTRY = str(ROOT / "app" / "app.py")


# --- scientific guardrails -------------------------------------------------------
def test_geometries_match_training_data_exactly():
    df = pd.read_csv(ROOT / "data/processed/openfoam_phase2_cleaned.csv")
    trained = set(
        df[df.airfoil_family == "biomimetic"][["tubercle_amplitude", "tubercle_wavelength"]]
        .itertuples(index=False, name=None)
    )
    offered = {(g["tubercle_amplitude"], g["tubercle_wavelength"]) for g in core.TUBERCLE_GEOMETRIES}
    assert offered == trained


def test_challenge_conditions_are_on_training_grid():
    df = pd.read_csv(ROOT / "data/processed/openfoam_phase2_cleaned.csv")
    c = core.CHALLENGE_CONDITIONS
    rows = df[(df.airfoil_family == c["airfoil_family"]) & (df.tubercle_shape == c["tubercle_shape"])
              & (df.angle_of_attack == c["angle_of_attack"]) & (df.airspeed == c["airspeed"])]
    assert len(rows) == len(core.TUBERCLE_GEOMETRIES)


def test_run_design_matches_backend_exactly():
    for i in range(len(core.TUBERCLE_GEOMETRIES)):
        rec = core.run_design(i)
        raw, clipped, was_clipped = predict_with_clipping(core.build_payload(i))
        # The saved ExtraTrees model uses n_jobs=-1, so parallel summation order can
        # differ in the last floating-point digit between calls (~1e-16).
        assert rec["raw"] == pytest.approx(raw, abs=1e-12)
        assert rec["x_over_c"] == pytest.approx(clipped, abs=1e-12)
        assert rec["was_clipped"] == was_clipped


def test_conditions_identical_across_designs():
    fixed = set(core.CHALLENGE_CONDITIONS)
    payloads = [core.build_payload(i) for i in range(len(core.TUBERCLE_GEOMETRIES))]
    assert all({k: p[k] for k in fixed} == core.CHALLENGE_CONDITIONS for p in payloads)


@pytest.mark.parametrize("bad", [-1, 6, 99])
def test_unsupported_geometry_rejected(bad):
    with pytest.raises(ValueError):
        core.build_payload(bad)


def test_comparison_threshold_classification():
    t = 0.02
    assert core.compare_to_start(0.81, 0.80, t)["verdict"] == "similar"
    assert core.compare_to_start(0.85, 0.80, t)["verdict"] == "later"
    assert core.compare_to_start(0.75, 0.80, t)["verdict"] == "earlier"


def test_explanations_make_no_performance_claims():
    banned = ["lift", "drag", "efficien", "emission", "optimal", "better"]
    for verdict, delta in [("later", 0.05), ("earlier", -0.05), ("similar", 0.0)]:
        text = core.explain_comparison(verdict, delta, 0.02).lower()
        assert not any(word in text for word in banned), (verdict, text)


def test_results_dataframe():
    start = core.run_design(0)
    df = core.results_dataframe(start, [core.run_design(2), core.run_design(5)])
    assert list(df["design"]) == ["starting design", "attempt 1", "attempt 2"]
    assert df.loc[0, "delta_vs_start"] == 0
    assert df["note"].str.contains("not a CFD result").all()


# --- app-level flows ---------------------------------------------------------------
def _app():
    at = AppTest.from_file(ENTRY, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_home_is_default_page():
    at = _app()
    assert at.title[0].value == "🛩️ WingCheck"


def test_wing_lab_full_challenge():
    at = _app()
    at.switch_page("wing_lab.py").run()
    assert not at.exception
    assert at.session_state["wl_start"]["geometry_index"] == core.STARTING_GEOMETRY_INDEX

    test_btn = lambda: next(b for b in at.button if b.label.startswith("🧪 Test My Wing"))  # noqa: E731
    for idx in (2, 4, 1):
        at.select_slider(key="wl_geometry").set_value(idx).run()
        test_btn().click().run()
        assert not at.exception
    assert [a["geometry_index"] for a in at.session_state["wl_attempts"]] == [2, 4, 1]
    assert test_btn().disabled  # attempt limit enforced

    # quiz
    at.radio(key="wl_quiz_choice").set_value(1).run()
    next(b for b in at.button if b.label == "Check my answer").click().run()
    assert at.success and at.success[0].value.startswith("Correct")

    # restart clears everything
    next(b for b in at.button if b.label.startswith("🔄")).click().run()
    assert at.session_state["wl_attempts"] == []


def test_advanced_mode_still_runs_and_predicts():
    at = _app()
    at.switch_page("advanced_mode.py").run()
    assert not at.exception
    next(b for b in at.button if b.label == "Launch WingCheck").click().run()
    next(b for b in at.button if b.label == "Start Here").click().run()
    at.button(key="FormSubmitter:input_form-Run Prediction").click().run()
    assert not at.exception
    assert at.session_state["latest_prediction"] is not None
    assert len(at.session_state["prediction_history"]) == 1
