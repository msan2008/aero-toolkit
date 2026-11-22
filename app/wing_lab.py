"""Wing Lab: a guided, beginner-friendly wing-design challenge.

All predictions go through wing_lab_core.run_design(), which calls the
existing src.inference.predict_with_clipping(). The UI uses native Streamlit
components plus one matplotlib chart, and matplotlib is already a dependency.
Challenge progress lives in st.session_state under keys prefixed "wl_", so it
never collides with Advanced Mode's state.
"""

import matplotlib.pyplot as plt
import streamlit as st

from src.inference import get_loaded_model_path, get_model_metrics
from wing_lab_core import (
    CHALLENGE_CONDITIONS,
    MAX_ATTEMPTS,
    STARTING_GEOMETRY_INDEX,
    TUBERCLE_GEOMETRIES,
    compare_to_start,
    comparison_threshold,
    explain_comparison,
    geometry_label,
    results_dataframe,
    run_design,
)

VERDICT_ICON = {"later": "➡️", "similar": "🟰", "earlier": "⬅️"}
VERDICT_TEXT = {
    "later": "Separation predicted later than the start",
    "similar": "About the same as the start",
    "earlier": "Separation predicted earlier than the start",
}

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
st.session_state.setdefault("wl_attempts", [])        # list of run_design() records
st.session_state.setdefault("wl_start", None)         # starting-design record
st.session_state.setdefault("wl_quiz_checked", False)
st.session_state.setdefault("wl_just_finished", False)


def restart_challenge() -> None:
    for key in [k for k in st.session_state.keys() if str(k).startswith("wl_")]:
        del st.session_state[key]


def test_my_wing() -> None:
    attempts = st.session_state.wl_attempts
    if len(attempts) >= MAX_ATTEMPTS:
        return
    attempts.append(run_design(st.session_state.wl_geometry))
    st.session_state.wl_just_finished = len(attempts) == MAX_ATTEMPTS


threshold = comparison_threshold()

# The starting design is predicted once, under the same fixed conditions.
if st.session_state.wl_start is None:
    try:
        st.session_state.wl_start = run_design(STARTING_GEOMETRY_INDEX)
    except Exception as exc:  # model missing or schema mismatch
        st.error(f"Wing Lab could not load the prediction model: {exc}")
        st.stop()

start = st.session_state.wl_start
attempts = st.session_state.wl_attempts
attempts_left = MAX_ATTEMPTS - len(attempts)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🐋 Wing Lab")
st.markdown(
    "**Your mission:** humpback whales have bumpy flippers. Engineers have copied those bumps "
    "onto wings. Can you tune the bumps so the model predicts the airflow stays attached "
    "**further back along the wing** than on the starting design?"
)
st.progress(len(attempts) / MAX_ATTEMPTS, text=f"Attempts used: {len(attempts)} of {MAX_ATTEMPTS}")

# -----------------------------------------------------------------------------
# Step 1: learn
# -----------------------------------------------------------------------------
st.header("1 · Learn the idea", divider="blue")
learn_left, learn_right = st.columns(2, gap="large")
with learn_left:
    st.subheader("Biomimicry")
    st.markdown(
        "Biomimicry means borrowing designs from nature. A humpback whale's flipper has "
        "rounded bumps called **tubercles** along its front edge. Wind-tunnel studies on "
        "flipper-shaped models have found that tubercles can change how airflow behaves at "
        "high angles, which is why engineers test them on wings."
    )
with learn_right:
    st.subheader("Flow separation")
    st.markdown(
        "Air flowing over a wing tries to follow its curved top surface. When the wing is "
        "tilted up steeply, the air can't keep up and **breaks away** from the surface. "
        "This is called *flow separation*. Behind that point the flow is messy and turbulent, "
        "and if it separates too early the wing can **stall**."
    )
st.markdown(
    "WingCheck predicts **where** separation starts, as a fraction of the chord "
    "(the distance from the front edge to the back edge of the wing):"
)
st.markdown(
    "- `x/c = 0.0` means separation right at the **front** (leading edge)\n"
    "- `x/c = 1.0` means the flow stays attached all the way to the **back** (trailing edge)"
)

# -----------------------------------------------------------------------------
# Step 2: the fixed test conditions
# -----------------------------------------------------------------------------
st.header("2 · The test conditions (locked)", divider="blue")
st.caption(
    "Every design is tested under exactly the same conditions, so the only thing that "
    "changes is the tubercle shape. Scientists call this a controlled comparison."
)
c1, c2, c3 = st.columns(3)
c1.metric("Wing type", "Biomimetic",
          help="A cambered wing with whale-style tubercles along its leading edge.")
c2.metric("Angle of attack", f"{CHALLENGE_CONDITIONS['angle_of_attack']:.0f}°",
          help="How steeply the wing is tilted into the oncoming air.")
c3.metric("Airspeed", f"{CHALLENGE_CONDITIONS['airspeed']} m/s",
          help="How fast the air moves past the wing.")

start_geo = TUBERCLE_GEOMETRIES[STARTING_GEOMETRY_INDEX]
st.info(
    f"**Starting design:** amplitude {start_geo['tubercle_amplitude']:.1f} mm, wavelength "
    f"{start_geo['tubercle_wavelength']:.1f} mm. Predicted separation at "
    f"**x/c = {start['x_over_c']:.2f}**. That's your number to compare against.",
    icon="🎯",
)

# -----------------------------------------------------------------------------
# Step 3: design and test
# -----------------------------------------------------------------------------
st.header("3 · Design your wing", divider="blue")
st.markdown(
    "- **Amplitude** is how tall each bump is.\n"
    "- **Wavelength** is how far apart the bumps are."
)
geometry_index = st.select_slider(
    "Choose a tubercle design",
    options=list(range(len(TUBERCLE_GEOMETRIES))),
    format_func=lambda i: f"Design {i + 1}",
    key="wl_geometry",
    disabled=attempts_left == 0,
    help="Only the six tubercle designs that were simulated in CFD are available.",
)
chosen = TUBERCLE_GEOMETRIES[geometry_index]
g1, g2 = st.columns(2)
g1.metric("Amplitude (bump height)", f"{chosen['tubercle_amplitude']:.1f} mm")
g2.metric("Wavelength (bump spacing)", f"{chosen['tubercle_wavelength']:.1f} mm")

with st.expander("Why can't I set amplitude and wavelength separately?"):
    st.markdown(
        "The model learned from CFD simulations of six specific tubercle designs, and in each "
        "one, amplitude and wavelength were changed **together** (taller bumps sit closer "
        "together). A model is only trustworthy for the kinds of inputs it learned from, so "
        "Wing Lab only offers those six designs. Asking it about other combinations would "
        "mean trusting a guess outside its experience."
    )

if any(a["geometry_index"] == geometry_index for a in attempts):
    st.caption("You've already tested this design. The model gives the same answer every time.")

st.button(
    f"🧪 Test My Wing ({attempts_left} {'attempt' if attempts_left == 1 else 'attempts'} left)",
    type="primary",
    on_click=test_my_wing,
    disabled=attempts_left == 0,
    use_container_width=True,
)

# -----------------------------------------------------------------------------
# Step 4: results
# -----------------------------------------------------------------------------
def plot_chord_indicator(start_rec, attempt_recs, rmse):
    """One horizontal chord per design. The marker is the predicted separation point.

    This is a 1-D plot of model outputs. It is not a simulated flow field.
    """
    labels = ["Start"] + [f"Attempt {i}" for i in range(1, len(attempt_recs) + 1)]
    values = [start_rec["x_over_c"]] + [a["x_over_c"] for a in attempt_recs]
    fig, ax = plt.subplots(figsize=(8, 0.75 * len(values) + 1.1))
    for row, (label, x) in enumerate(zip(labels, values)):
        y = len(values) - 1 - row
        colour = "#8a8f98" if row == 0 else "#2178a8"
        ax.barh(y, x, height=0.45, color=colour, alpha=0.85)                        # attached
        ax.barh(y, 1 - x, left=x, height=0.45, color="#e9ecef", hatch="///",
                edgecolor="#b0b5bd", linewidth=0)                                  # separated
        ax.errorbar(x, y, xerr=rmse, fmt="o", color="black", capsize=4, markersize=6)
        ax.text(1.01, y, f"x/c = {x:.2f}", va="center", fontsize=10)
    ax.axvline(start_rec["x_over_c"], color="#8a8f98", linestyle=":", linewidth=1.2)
    ax.set_yticks(range(len(values)), labels[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Position along the chord (x/c):  0 = leading edge  →  1 = trailing edge")
    ax.set_title("Predicted separation point (solid = attached flow, hatched = separated)",
                 fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


if attempts:
    st.header("4 · Your results", divider="blue")
    if st.session_state.wl_just_finished:
        st.balloons()
        st.session_state.wl_just_finished = False

    latest = attempts[-1]
    latest_cmp = compare_to_start(latest["x_over_c"], start["x_over_c"], threshold)
    r1, r2 = st.columns(2)
    r1.metric(
        f"Attempt {len(attempts)}: predicted separation",
        f"x/c = {latest['x_over_c']:.2f}",
        delta=f"{latest_cmp['delta']:+.3f} vs start",
        delta_color="off",  # grey: a later separation point is not automatically "good"
    )
    r2.metric("Starting design", f"x/c = {start['x_over_c']:.2f}")
    st.markdown(
        f"**What this means:** the model predicts the air stays attached over roughly the first "
        f"**{latest['x_over_c'] * 100:.0f}%** of this wing's chord, then separates. "
        + explain_comparison(latest_cmp["verdict"], latest_cmp["delta"], threshold)
    )
    if latest["was_clipped"]:
        st.warning(
            f"The model's raw output ({latest['raw']:.3f}) fell outside 0–1 and was clipped. "
            "Treat this result as unreliable."
        )

    fig = plot_chord_indicator(start, attempts, threshold)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"The black error bars show the model's typical error (±{threshold:.2f} x/c). If two "
        "error bars overlap a lot, the model can't confidently tell those designs apart. "
        "This chart only marks one predicted number per design. It is **not** a simulation "
        "or picture of the actual airflow."
    )

    st.subheader("Attempt log")
    for i, rec in enumerate(attempts, start=1):
        cmp = compare_to_start(rec["x_over_c"], start["x_over_c"], threshold)
        st.markdown(
            f"{VERDICT_ICON[cmp['verdict']]} **Attempt {i}**, {geometry_label(rec['geometry_index'])} "
            f"→ x/c = {rec['x_over_c']:.2f} ({cmp['delta']:+.3f}): {VERDICT_TEXT[cmp['verdict']]}"
        )

    st.warning(
        "**Remember:** a later separation point is *one* clue about a wing. It does not by itself "
        "mean more lift, less drag, lower energy use, or a better wing overall. The model also "
        "only describes these exact test conditions.",
        icon="⚠️",
    )

# -----------------------------------------------------------------------------
# Step 5: wrap-up question (after all attempts)
# -----------------------------------------------------------------------------
QUIZ_OPTIONS = [
    "Yes. Later separation always means the wing is better.",
    "Not necessarily. Separation is only one factor, and the model only tested one angle and speed.",
    "No. A later separation point always means more drag.",
]
QUIZ_CORRECT = 1
QUIZ_FEEDBACK = [
    "Not quite. Delaying separation is often helpful, but a real wing is also judged on lift, "
    "drag, weight, strength, and how it behaves across many angles and speeds. This model "
    "predicts none of those.",
    "Correct! Separation location is one useful clue, but engineers also need lift, drag, "
    "structural, and off-design behaviour, and they confirm results with CFD or wind tunnels. "
    "Your test only covered 12° at 15 m/s.",
    "Not quite. Later separation doesn't *always* mean more drag. The real point is that this "
    "model predicts separation location only, so it can't tell you about drag at all.",
]

if len(attempts) >= MAX_ATTEMPTS:
    st.header("5 · Think like an engineer", divider="blue")
    furthest = max(range(len(attempts)), key=lambda i: attempts[i]["x_over_c"])
    st.markdown(
        f"Among your attempts, **Attempt {furthest + 1}** had the furthest-back predicted "
        f"separation (x/c = {attempts[furthest]['x_over_c']:.2f})."
    )
    choice = st.radio(
        "If one design has the latest predicted separation point, can we say it is the best wing?",
        options=range(len(QUIZ_OPTIONS)),
        format_func=lambda i: QUIZ_OPTIONS[i],
        index=None,
        key="wl_quiz_choice",
        on_change=lambda: st.session_state.update(wl_quiz_checked=False),
    )
    if st.button("Check my answer", disabled=choice is None):
        st.session_state.wl_quiz_checked = True
    if st.session_state.wl_quiz_checked and choice is not None:
        (st.success if choice == QUIZ_CORRECT else st.info)(QUIZ_FEEDBACK[choice])

# -----------------------------------------------------------------------------
# Restart / download
# -----------------------------------------------------------------------------
st.divider()
b1, b2 = st.columns(2)
with b1:
    st.button("🔄 Restart challenge", on_click=restart_challenge, use_container_width=True)
with b2:
    st.download_button(
        "⬇️ Download my results (CSV)",
        data=results_dataframe(start, attempts).to_csv(index=False).encode("utf-8"),
        file_name="wing_lab_results.csv",
        mime="text/csv",
        disabled=not attempts,
        use_container_width=True,
    )

# -----------------------------------------------------------------------------
# Reliability and limitations
# -----------------------------------------------------------------------------
with st.expander("About the model and its limits"):
    metrics = get_model_metrics()
    mae = f"{metrics['mae']:.3f} x/c" if "mae" in metrics else "N/A"
    r2 = f"{metrics['r2']:.3f}" if "r2" in metrics else "N/A"
    st.markdown(
        f"- **Model file:** `{get_loaded_model_path().split('/')[-1]}` (Extra Trees regressor)\n"
        "- **Trained on:** about 1,700 OpenFOAM CFD simulations (Phase 2 dataset)\n"
        f"- **Held-out test error:** MAE {mae}, R² {r2}\n"
        "- **Predicts:** separation location `separation_x_over_c` only\n"
        "- **Does not predict:** lift, drag, efficiency, battery life, or emissions\n"
        "- **Valid for:** the six simulated tubercle designs, angles 0–24°, airspeeds 15 or 30 m/s\n"
        "- This is an educational screening prototype, not a replacement for CFD, wind-tunnel "
        "testing, or professional aerodynamic design."
    )
