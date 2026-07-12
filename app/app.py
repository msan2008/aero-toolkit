import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Robust path setup so Streamlit can import from src/ whether this file is run
# from app/app.py or from the project root as app.py.
# -----------------------------------------------------------------------------
CURRENT_FILE = Path(__file__).resolve()
CANDIDATE_ROOTS = [CURRENT_FILE.parent, CURRENT_FILE.parent.parent]
PROJECT_ROOT = next((root for root in CANDIDATE_ROOTS if (root / "src").exists()), CURRENT_FILE.parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    describe_prediction,
    get_required_feature_columns,
    predict_from_dict,
)

# predict_raw_from_dict() returns the UNCLIPPED model output, which is what lets
# this app distinguish a genuine 1.0 from a clamped 1.4. It only exists in the
# updated src/inference.py; if the app is run against an older copy, we fall
# back to the clipped value and detect clipping from the boundary instead.
try:
    from src.inference import predict_raw_from_dict  # noqa: E402

    HAS_RAW_PREDICTION = True
except ImportError:
    HAS_RAW_PREDICTION = False

# -----------------------------------------------------------------------------
# Theme accent (self-contained, no .streamlit/config.toml required)
#
# Streamlit's default primaryColor is #FF4B4B, a red that drives the toggle
# fills, primary buttons, slider handles, and focus rings. Normally you would
# change it in .streamlit/config.toml. To keep everything in this single file,
# we set the same option at runtime. This touches a semi-private API, so it is
# wrapped in try/except: if it fails on a given Streamlit version, the CSS
# block further down still repaints the visible controls.
# -----------------------------------------------------------------------------
ACCENT = "#3f6184"          # slate blue — used for primary buttons (needs
ACCENT_HOVER = "#34506d"    # enough contrast for white button text)
TRACK_OFF = "#c8ccd4"       # unchecked toggle track (light mode)

# Baby-blue palette for the sidebar, toggles, and sliders. Three shades so the
# on-state toggle stays visible against the sidebar it sits on:
#   SIDEBAR_BLUE  soft, pale — the sidebar background
#   BABY_BLUE     stronger fill — toggle "on" track, slider thumb + filled track
#   BABY_BLUE_TEXT deeper — slider numbers/ticks, readable on white
SIDEBAR_BLUE = "#eef3fb"    # white with a faint blue tint (sidebar background)
BABY_BLUE = "#7fc4ee"
BABY_BLUE_TEXT = "#2178a8"

# How close to 0.0 or 1.0 counts as "sitting on the boundary" (i.e. probably
# already clamped somewhere upstream) rather than a genuine interior prediction.
CLIP_EPS = 1e-6


def _apply_theme_options() -> None:
    try:
        from streamlit import config as _st_config

        # primaryColor drives the slider's *filled* track and thumb, the toggle
        # on-state, and focus rings — so baby blue here colors the sliders and
        # toggles even where CSS can't reliably reach the filled track. Primary
        # buttons are forced back to slate by the CSS below so their white text
        # stays legible.
        _st_config.set_option("theme.primaryColor", BABY_BLUE)
        _st_config.set_option("theme.secondaryBackgroundColor", SIDEBAR_BLUE)
    except Exception:
        # Older/newer Streamlit, or config locked after server start.
        # The CSS overrides below cover the visible surfaces.
        pass


_apply_theme_options()

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="WingCheck",
    page_icon="🛩️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Lightweight styling
# -----------------------------------------------------------------------------
# The accent is exposed to CSS as custom properties so the Python constants
# above remain the single source of truth for the color.
st.markdown(
    f"""
    <style>
    :root {{
        --aero-accent: {ACCENT};
        --aero-accent-hover: {ACCENT_HOVER};
        --aero-track-off: {TRACK_OFF};
        --aero-sidebar: {SIDEBAR_BLUE};
        --aero-baby: {BABY_BLUE};
        --aero-baby-text: {BABY_BLUE_TEXT};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Base size bumped from the 16px default to 19px (+3pt). Because Streamlit
       sizes most text in rem, this scales the whole type hierarchy up together
       while preserving the relative sizes of headings vs. body text. */
    html { font-size: 19px; }

    html, body, [class*="css"], .stApp,
    .stMarkdown, .stText,
    h1, h2, h3, h4, h5, h6, p, span, label, div,
    button, input, select, textarea,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        font-family: 'Poppins', system-ui, sans-serif !important;
        color: #000000 !important;
    }

    [data-testid="stIconMaterial"],
    span[class*="material-icons"],
    [class*="material-symbols"],
    .material-icons,
    .material-icons-outlined {
        font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 9rem;
        padding-right: 6rem;
        max-width: 1040px;
        margin: 0 auto;
    }
    .info-card {
        padding: 1rem 1.15rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(49, 51, 63, 0.15);
        background: rgba(240, 242, 246, 0.45);
        margin-bottom: 0.75rem;
    }
    .small-note {
        font-size: 0.92rem;
        opacity: 0.82;
    }

    /* Force the sidebar to be smaller. White with a faint blue tint; the
       toggle "on" color is stronger so the toggles stay visible against it. */
    [data-testid="stSidebar"] {
        background-color: var(--aero-sidebar);
        min-width: 220px !important;
        max-width: 220px !important;
    }

    /* Sidebar toggle labels: single-word labels plus a slightly smaller size so
       they never wrap onto a second line at the 220px sidebar width. */
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label p {
        font-size: 0.9rem !important;
        white-space: nowrap;
    }

    /* ---- Baby-blue toggles ----
       The toggle track is a span in some Streamlit versions and a div in
       others, so both are matched. aria-checked distinguishes on/off. */
    [data-baseweb="checkbox"] span[aria-checked="true"],
    [data-baseweb="checkbox"] div[aria-checked="true"],
    [data-testid="stCheckbox"] [aria-checked="true"] {
        background-color: var(--aero-baby) !important;
    }
    [data-baseweb="checkbox"] span[aria-checked="false"],
    [data-baseweb="checkbox"] div[aria-checked="false"],
    [data-testid="stCheckbox"] [aria-checked="false"] {
        background-color: var(--aero-track-off) !important;
    }
    /* The knob itself stays white on both tracks. */
    [data-baseweb="checkbox"] [aria-checked] > div {
        background-color: #ffffff !important;
    }
    [data-baseweb="checkbox"] input:focus + div,
    [data-baseweb="checkbox"] [aria-checked]:focus-visible {
        box-shadow: 0 0 0 3px rgba(127, 196, 238, 0.45) !important;
    }

    /* ---- Baby-blue sliders (AoA, amplitude, wavelength, carbon) ---- */
    [data-baseweb="slider"] [role="slider"] {
        background-color: var(--aero-baby) !important;
        border-color: var(--aero-baby) !important;
    }
    [data-baseweb="slider"] [data-testid="stThumbValue"],
    [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {
        color: var(--aero-baby-text) !important;
    }

    /* ---- Neutral accent for primary buttons ---- */
    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"],
    [data-testid="stFormSubmitButton"] > button {
        background-color: var(--aero-accent) !important;
        border-color: var(--aero-accent) !important;
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"] *,
    .stFormSubmitButton > button[kind="primary"] *,
    [data-testid="stFormSubmitButton"] > button * {
        color: #ffffff !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button[kind="primary"]:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        background-color: var(--aero-accent-hover) !important;
        border-color: var(--aero-accent-hover) !important;
    }

    /* Secondary buttons ("Clear history", CSV download) turn red on hover by
       default; keep them on the same neutral accent. */
    .stButton > button:not([kind="primary"]):hover,
    .stDownloadButton > button:hover {
        border-color: var(--aero-accent) !important;
        color: var(--aero-accent) !important;
    }
    .stButton > button:not([kind="primary"]):hover *,
    .stDownloadButton > button:hover * {
        color: var(--aero-accent) !important;
    }

    /* Links and focus rings */
    a, a:visited { color: var(--aero-accent) !important; }
    *:focus-visible { outline-color: var(--aero-accent) !important; }

    /* Condense the labels in the top input bar */
    .condensed-label label {
        font-size: 0.85rem !important;
        margin-bottom: 0px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def get_model_metrics_safe() -> dict[str, float]:
    """Return held-out validation metrics, or {} if none are recorded.

    Falls back to the R^2-only API when running against an older
    src/inference.py that predates get_model_metrics().
    """
    try:
        from src.inference import get_model_metrics  # type: ignore  # noqa: E402

        return {k: float(v) for k, v in get_model_metrics().items() if v is not None}
    except Exception:
        score = get_model_r2()
        return {"r2": score} if score is not None else {}


# Display order used by the metric dropdown and the reliability note.
METRIC_KEYS = ("r2", "mae", "rmse")

METRIC_LABELS = {
    "r2": "Model R²",
    "mae": "Model MAE",
    "rmse": "Model RMSE",
}

# One-line plain-language reading of each metric, shown under the selected value.
METRIC_HELP = {
    "r2": "Share of test-set variation the model explains (1.00 is perfect). Higher is better.",
    "mae": "Average error on held-out designs, in chord fraction. Lower is better.",
    "rmse": "Like MAE but penalizes large misses more, in chord fraction. Lower is better.",
}


def format_metric_value(metric_key: str, value: float) -> str:
    """R² is a bare 0–1 score; MAE/RMSE are chord fractions, shown with a unit."""
    if metric_key == "r2":
        return f"{value:.3f}"
    return f"{value:.3f} x/c"


def format_reliability_note(metrics: dict[str, float]) -> str:
    """Build a one-line held-out performance summary from whatever is available."""
    parts = []
    if "r2" in metrics:
        parts.append(f"R² = {metrics['r2']:.3f}")
    if "mae" in metrics:
        parts.append(f"MAE = {metrics['mae']:.3f} x/c")
    if "rmse" in metrics:
        parts.append(f"RMSE = {metrics['rmse']:.3f} x/c")

    note = "**Model reliability (held-out test set):** " + " · ".join(parts)

    # MAE is in units of chord fraction, so it converts directly into the
    # "how wrong is this number likely to be" statement users actually want.
    if "mae" in metrics:
        note += (
            f". On unseen designs the model is typically off by about "
            f"{metrics['mae'] * 100:.1f}% of the chord, so read the prediction above "
            "as a range, not a point value."
        )
    return note


def get_model_r2() -> float | None:
    """Return the trained model's R^2 score if the inference layer exposes one.

    This tries to read the value from src/inference.py so the displayed score
    stays in sync with the saved model. If no score is available, it returns
    None and the UI shows "N/A" instead of crashing.
    """
    try:
        from src.inference import get_model_r2_score  # type: ignore  # noqa: E402

        score = get_model_r2_score()
        return float(score) if score is not None else None
    except Exception:
        return None


def evaluate_clipping(raw_prediction: float) -> tuple[float, str | None]:
    """Clamp a model output to the physical range [0, 1] and report why.

    `separation_x_over_c` is a normalized chord position, so values outside
    [0, 1] are physically meaningless — a regressor is free to produce them
    under extrapolation.

    Returns (bounded_value, status). Status is None for a clean interior
    prediction, otherwise:

      "clamped_low"/"clamped_high"  : the raw output was outside [0, 1].
      "at_lower_bound"/"at_upper_bound"
          : the value is exactly on a boundary. When the raw output is
            available this is a genuine saturated prediction; when it is not
            (older src/inference.py, which clips internally), it means the
            value was clamped upstream and the true output is unknown.
    """
    if raw_prediction < 0.0:
        return 0.0, "clamped_low"
    if raw_prediction > 1.0:
        return 1.0, "clamped_high"
    if raw_prediction <= CLIP_EPS:
        return raw_prediction, "at_lower_bound"
    if raw_prediction >= 1.0 - CLIP_EPS:
        return raw_prediction, "at_upper_bound"
    return raw_prediction, None


def clipping_message(status: str, raw_prediction: float) -> str:
    """Explain a clipping status in workshop-appropriate language."""
    if status == "clamped_high":
        return (
            f"The model returned {raw_prediction:.3f}, which would place separation past the "
            "trailing edge. It has been clamped to 1.00. Read this as "
            "\"no separation predicted within the chord\" — the inputs pushed the model outside "
            "the range it was trained on, so the number is not a precise result."
        )
    if status == "clamped_low":
        return (
            f"The model returned {raw_prediction:.3f}, which would place separation ahead of the "
            "leading edge. It has been clamped to 0.00. Read this as "
            "\"separation predicted immediately\" — the inputs pushed the model outside the range "
            "it was trained on, so the number is not a precise result."
        )

    edge = "upper bound (x/c = 1.00)" if status == "at_upper_bound" else "lower bound (x/c = 0.00)"
    if HAS_RAW_PREDICTION:
        return (
            f"The prediction sits exactly on the {edge}. The model saturated at the edge of the "
            "physical range, so treat this as a boundary case rather than a confident estimate."
        )
    return (
        f"The prediction sits exactly on the {edge}. This build of src/inference.py clips before "
        "the value reaches the interface, so the underlying model output is unknown and may lie "
        "well past the edge. This is a boundary artifact, not a confident prediction."
    )


def plot_airfoil_and_separation(
    separation_x_over_c: float,
    dark_mode: bool = False,
    baseline_x_over_c: float | None = None,
) -> plt.Figure:
    if dark_mode:
        bg_color = "#0e1117"
        fg_color = "#e6e6e6"
        line_color = "#5aa9e6"
        marker_color = "#ff7b72"
    else:
        bg_color = "white"
        fg_color = "black"
        line_color = None  # use matplotlib defaults to keep light mode unchanged
        marker_color = None

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    ax.plot([0, 1], [0, 0], linewidth=4, color=line_color)
    ax.scatter([separation_x_over_c], [0], s=120, zorder=3, color=marker_color)
    ax.axvline(separation_x_over_c, linestyle="--", linewidth=1.5, color=marker_color)
    ax.text(
        separation_x_over_c,
        0.08,
        f"x/c = {separation_x_over_c:.2f}",
        ha="center",
        va="bottom",
        fontsize=11,
        color=fg_color,
    )

    ax.text(0.0, -0.11, "Leading edge", ha="left", va="top", fontsize=9, color=fg_color)
    ax.text(1.0, -0.11, "Trailing edge", ha="right", va="top", fontsize=9, color=fg_color)

    # Optional symmetric-baseline reference, drawn muted so the selected design
    # stays visually dominant.
    if baseline_x_over_c is not None:
        baseline_color = "#8a8f98" if not dark_mode else "#9aa0a8"
        ax.axvline(baseline_x_over_c, linestyle=":", linewidth=1.5, color=baseline_color)
        ax.scatter([baseline_x_over_c], [0], s=70, zorder=2, color=baseline_color, marker="s")
        ax.text(
            baseline_x_over_c,
            -0.20,
            f"baseline {baseline_x_over_c:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=baseline_color,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 0.35)
    ax.set_xlabel("Normalized chord location (x/c)", color=fg_color)
    ax.set_yticks([])
    ax.set_title("Predicted Flow Separation Location Along Wing Chord", color=fg_color)
    ax.tick_params(axis="x", colors=fg_color)
    for spine in ax.spines.values():
        spine.set_color(fg_color)
    ax.grid(True, axis="x", alpha=0.3)
    return fig


def run_single_prediction(payload: dict) -> dict:
    """Predict once and return the rounded value plus clipping provenance."""
    if HAS_RAW_PREDICTION:
        model_output = float(predict_raw_from_dict(payload))
    else:
        model_output = float(predict_from_dict(payload))
    bounded, clip_status = evaluate_clipping(model_output)
    return {
        "prediction": round(bounded, 2),
        "raw": model_output,
        "clip_status": clip_status,
    }


def build_symmetric_baseline_inputs(payload: dict) -> dict:
    """Return the same flow condition on a plain symmetric wing.

    Airspeed, angle of attack, chords, and sweep are held fixed; only the
    airfoil family and the tubercle geometry change. That makes the comparison
    a like-for-like question: at this exact flow condition, what does the
    tubercle geometry buy you?
    """
    baseline = dict(payload)
    baseline["airfoil_family"] = "symmetric"
    baseline["tubercle_shape"] = "none"
    baseline["tubercle_amplitude"] = 0.0
    baseline["tubercle_wavelength"] = 0.0
    return baseline


def describe_baseline_delta(delta: float) -> str:
    """Plain-language reading of (biomimetic - symmetric) separation location."""
    # Below one rounding unit, the two predictions are indistinguishable at the
    # precision this app displays. Claiming a difference would be false precision.
    if abs(delta) < 0.01:
        return (
            "The model predicts effectively the same separation location for both wings "
            "at this flow condition — no meaningful difference at the precision shown."
        )
    if delta > 0:
        return (
            f"The model predicts separation about {abs(delta) * 100:.0f}% of the chord **later** "
            "on the biomimetic wing than on a symmetric wing at the same angle of attack and "
            "airspeed. Later separation is the outcome tubercles are intended to produce."
        )
    return (
        f"The model predicts separation about {abs(delta) * 100:.0f}% of the chord **earlier** "
        "on the biomimetic wing than on a symmetric wing at the same angle of attack and "
        "airspeed. At this flow condition the tubercle geometry is not helping."
    )


def sustainability_interpretation(separation_x_over_c: float) -> tuple[str, str]:
    if separation_x_over_c >= 0.80:
        return (
            "Strong attached-flow indicator",
            "The predicted separation point is far back on the chord. In a real design cycle, this would be a promising candidate for follow-up CFD or wind-tunnel testing because delayed separation can support more efficient flight.",
        )
    if separation_x_over_c >= 0.60:
        return (
            "Moderate attached-flow indicator",
            "The predicted separation point is in a reasonable range. This design may be worth comparing against nearby designs with small changes to angle of attack, airspeed, or tubercle geometry.",
        )
    return (
        "Early-separation warning",
        "The predicted separation point is relatively far forward. This may indicate a less efficient design that could waste more energy through separated flow and should be improved before more expensive testing.",
    )


def build_sustainability_table(
    energy_per_flight_wh: float,
    number_of_flights: int,
    carbon_factor_kg_per_kwh: float,
) -> pd.DataFrame:
    scenarios = [2, 5, 10]
    rows = []
    total_energy_kwh = (energy_per_flight_wh * number_of_flights) / 1000.0
    for efficiency_gain_percent in scenarios:
        saved_kwh = total_energy_kwh * (efficiency_gain_percent / 100.0)
        avoided_kg_co2 = saved_kwh * carbon_factor_kg_per_kwh
        rows.append(
            {
                "Assumed efficiency improvement": f"{efficiency_gain_percent}%",
                "Hypothetical energy saved (kWh)": round(saved_kwh, 3),
                "Estimated CO₂ avoided under assumed efficiency gain (kg)": round(avoided_kg_co2, 3),
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Welcome page
# -----------------------------------------------------------------------------
if "entered" not in st.session_state:
    st.session_state.entered = False

if not st.session_state.entered:
    st.markdown("<div style='height: 6vh;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<h1 style='text-align: center; font-size: 3rem;'>Welcome to WingCheck</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;' class='small-note'>"
        "A lightweight machine-learning tool for screening biomimetic wing designs "
        "before committing to CFD or wind-tunnel testing."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;' class='small-note'>"
        "The app predicts where airflow separation may occur along a wing chord, "
        "helping users compare early-stage aerodynamic designs."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 2vh;'></div>", unsafe_allow_html=True)

    spacer_left, center_col, spacer_right = st.columns([1, 2, 1])
    with center_col:
        if st.button(
            "Launch WingCheck",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.entered = True
            st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# App title and overview
# -----------------------------------------------------------------------------
st.title("WingCheck: Biomimetic Wing Screening Tool")
st.markdown(
    """
    This prototype uses a saved machine learning model to predict the flow separation
    location, `separation_x_over_c`, from wing geometry and flow inputs. The goal is
    to help students, makers, and robotics teams quickly screen wing ideas before
    committing to more expensive CFD or wind-tunnel testing.
    """
)

# -----------------------------------------------------------------------------
# Presets
#
# Each preset writes directly into the widget keys used by the input bar below.
# Tubercle values are always stored inside the sliders' valid ranges (even for
# the non-biomimetic presets) so the sliders never re-render with an
# out-of-bounds value; the "none"/0.0 substitution happens later, when the
# input dictionary is assembled.
# -----------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "Symmetric baseline": {
        "airfoil_family": "symmetric",
        "angle_of_attack": 5.0,
        "airspeed": 15,
        "tubercle_shape": "whale",
        "tubercle_amplitude": 26.2,
        "tubercle_wavelength": 49.6,
    },
    "Cambered baseline": {
        "airfoil_family": "cambered",
        "angle_of_attack": 5.0,
        "airspeed": 15,
        "tubercle_shape": "whale",
        "tubercle_amplitude": 26.2,
        "tubercle_wavelength": 49.6,
    },
    "Biomimetic default": {
        "airfoil_family": "biomimetic",
        "angle_of_attack": 10.0,
        "airspeed": 30,
        "tubercle_shape": "whale",
        "tubercle_amplitude": 26.2,
        "tubercle_wavelength": 49.6,
    },
    # A deliberately hard case: a plain symmetric section at a high angle of
    # attack and low airspeed. Separation should be predicted early, giving
    # workshop participants a design to improve — the intended move is to
    # switch the family to biomimetic and tune the tubercle geometry.
    "Workshop challenge": {
        "airfoil_family": "symmetric",
        "angle_of_attack": 20.0,
        "airspeed": 15,
        "tubercle_shape": "whale",
        "tubercle_amplitude": 32.7,
        "tubercle_wavelength": 42.3,
    },
}

# Biomimetic is the default because the project is centered on biomimicry, and
# workshop users should see the tubercle controls on first load. Change this
# one string to "Symmetric baseline" to start from the flat-plate comparison.
DEFAULT_PRESET = "Biomimetic default"

if "active_preset" not in st.session_state:
    st.session_state.active_preset = DEFAULT_PRESET

# setdefault (not direct assignment) so a user's manual edits survive reruns.
# It also self-heals: Streamlit drops session_state entries for widgets that
# were not rendered on the previous run, which is what happens to the tubercle
# sliders whenever a non-biomimetic family is selected.
for _key, _value in PRESETS[DEFAULT_PRESET].items():
    st.session_state.setdefault(_key, _value)


def apply_preset(name: str) -> None:
    """Write a preset into the widget keys.

    This runs before the input widgets are instantiated on the current script
    run, so no explicit st.rerun() is needed: the button click already
    triggered the rerun, and the widgets pick these values up as they render.
    """
    for key, value in PRESETS[name].items():
        st.session_state[key] = value
    st.session_state.active_preset = name


# -----------------------------------------------------------------------------
# Top Input Bar (Condensed Single Row, batched inside a form)
# -----------------------------------------------------------------------------
st.markdown("---")

root_chord = 1.0
tip_chord = 1.0
sweep_angle = 0.0

# Preset buttons must live outside the form: Streamlit only permits
# st.form_submit_button inside a form block.
preset_cols = st.columns(len(PRESETS))
for _col, _preset_name in zip(preset_cols, PRESETS):
    with _col:
        if st.button(_preset_name, use_container_width=True, key=f"preset_{_preset_name}"):
            apply_preset(_preset_name)

st.caption(
    f"Last preset applied: **{st.session_state.active_preset}**. "
    "Presets fill the input bar below — you can still adjust any control before running."
)

# Wrapping the controls in a form means widget changes (especially dragging the
# AoA slider) no longer trigger a rerun on every interaction. The script only
# reruns when the user clicks the submit button below.
with st.form("input_form"):
    # Wrapper to apply tighter margins/labels via our custom CSS
    st.markdown('<div class="condensed-label">', unsafe_allow_html=True)

    # 7 columns for a highly condensed, single-bar layout
    cols = st.columns([1.2, 1, 1, 0.8, 1.2, 1, 1])

    # Every widget below is driven by session_state via `key`, which is what
    # lets the preset buttons populate them. Note that `value=`/`index=` are
    # deliberately omitted: passing both a key and a default triggers a
    # Streamlit warning about conflicting sources of truth.
    with cols[0]:
        airfoil_family = st.selectbox(
            "Airfoil Family",
            ["symmetric", "cambered", "biomimetic"],
            key="airfoil_family",
        )
    with cols[1]:
        angle_of_attack = st.slider(
            "AoA (°)", min_value=0.0, max_value=25.0, step=0.5, format="%.1f",
            key="angle_of_attack",
        )
    with cols[2]:
        airspeed = st.selectbox(
            "Airspeed (m/s)",
            [15, 30],
            format_func=lambda v: f"{v} m/s",
            key="airspeed",
        )
    with cols[3]:
        st.markdown(
            "<div style='font-size:0.8rem; margin-top:0.4rem; opacity:0.8;'>"
            "The sweep angle, root chord, and tip chord are fixed</div>",
            unsafe_allow_html=True,
        )

    if airfoil_family == "biomimetic":
        with cols[4]:
            tubercle_shape = st.selectbox(
                "Tubercle Shape", ["whale", "biomimetic_v1"], key="tubercle_shape"
            )
        with cols[5]:
            tubercle_amplitude = st.slider(
                "Amplitude (mm)", min_value=26.2, max_value=32.7, step=0.1,
                format="%.1f mm", key="tubercle_amplitude",
            )
        with cols[6]:
            tubercle_wavelength = st.slider(
                "Wavelength (mm)", min_value=42.3, max_value=49.6, step=0.1,
                format="%.1f mm", key="tubercle_wavelength",
            )
    else:
        tubercle_shape = "none"
        tubercle_amplitude = 0.0
        tubercle_wavelength = 0.0

    st.markdown('</div>', unsafe_allow_html=True)
    st.caption("Inputs are limited to the model's training range to avoid unsupported extrapolation.")
    submitted = st.form_submit_button("Run Prediction", type="primary")

st.markdown("---")

input_dict = {
    "airfoil_family": airfoil_family,
    "tubercle_amplitude": tubercle_amplitude,
    "tubercle_wavelength": tubercle_wavelength,
    "tubercle_shape": tubercle_shape,
    "root_chord": root_chord,
    "tip_chord": tip_chord,
    "sweep_angle": sweep_angle,
    "angle_of_attack": angle_of_attack,
    "airspeed": airspeed,
}

# -----------------------------------------------------------------------------
# Sidebar Display Options
# -----------------------------------------------------------------------------
st.sidebar.header("Display Options")
show_explanations = st.sidebar.toggle("Explanations", value=False)
show_prediction = st.sidebar.toggle("Prediction", value=True)
show_sustainability = st.sidebar.toggle("Sustainability", value=False)
show_about = st.sidebar.toggle("About", value=False)
dark_mode = st.sidebar.toggle("Dark Mode", value=False)

# Inject dark-theme overrides. This is rendered after the base style block, so
# its rules win the cascade for the selectors they share.
if dark_mode:
    st.markdown(
        """
        <style>
        /* ---- Aero Toolkit dark mode ---- */
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(14, 17, 23, 0) !important;
        }

        /* Light text overrides the forced-black base rule */
        html, body, [class*="css"], .stApp,
        .stMarkdown, .stText,
        h1, h2, h3, h4, h5, h6, p, span, label, div,
        button, input, select, textarea,
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {
            color: #e6e6e6 !important;
        }

        /* Sidebar: black in dark mode (overrides the light-mode blue tint). */
        [data-testid="stSidebar"] {
            background-color: #0e1117 !important;
        }
        /* Dark sidebar needs light text again. */
        [data-testid="stSidebar"] * {
            color: #e6e6e6 !important;
        }
        /* Sidebar toggle OFF track: a darker neutral reads clearly on black. */
        [data-testid="stSidebar"] [data-baseweb="checkbox"] [aria-checked="false"] {
            background-color: #3a3f4a !important;
        }

        /* Toggle track (off state) needs a darker neutral on dark backgrounds */
        [data-baseweb="checkbox"] [aria-checked="false"] {
            background-color: #3a3f4a !important;
        }
        [data-baseweb="checkbox"] [aria-checked="true"] {
            background-color: #5b86b3 !important;
        }

        /* Primary buttons keep white label text on the accent fill */
        .stButton > button[kind="primary"] *,
        .stFormSubmitButton > button[kind="primary"] * {
            color: #ffffff !important;
        }

        /* Info cards */
        .info-card {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
        }

        /* Inputs / text areas keep the dark fill with light text. */
        input, textarea,
        [data-baseweb="input"] > div,
        [data-baseweb="base-input"] {
            background-color: #262730 !important;
            color: #e6e6e6 !important;
        }

        /* ---- Captions / helper text ----
           Streamlit renders captions as dimmed grey, which is hard to read on
           a dark background. Force them to full-opacity white. */
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] *,
        .stCaption, .stCaption *,
        small, .small-note {
            color: #f5f5f5 !important;
            opacity: 1 !important;
        }

        /* ---- Select / dropdown boxes (closed control) ----
           Requested black text. Black is only legible on a light fill, so the
           control gets a light background. BaseWeb nests the visible value a
           few levels deep and sets its own colors inline, so we cast a wide net
           and force every descendant dark. */
        [data-baseweb="select"],
        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div > div,
        [data-baseweb="select"] div[role="button"],
        [data-baseweb="select"] [data-baseweb="base-input"] {
            background-color: #eef4fa !important;
        }
        [data-baseweb="select"] *,
        [data-baseweb="select"] div[value],
        [data-baseweb="select"] span,
        [data-baseweb="select"] input {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
        }
        /* The dropdown chevron/clear icons, kept dark so they show on the light fill. */
        [data-baseweb="select"] svg {
            fill: #111111 !important;
            color: #111111 !important;
        }

        /* ---- Open dropdown menu ----
           The popover is portaled to the <body>, outside stApp, so it is NOT
           covered by the dark theme's text rules and needs its own. Each option
           must be forced dark (BaseWeb dims non-highlighted options via inline
           opacity/color) and the highlighted option needs a visible tint so it
           is not white text on a near-white row. */
        [data-baseweb="popover"] ul,
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="menu"] li,
        [role="listbox"] [role="option"] {
            background-color: #ffffff !important;
        }
        [data-baseweb="popover"] li,
        [data-baseweb="popover"] li *,
        [data-baseweb="popover"] [role="option"],
        [data-baseweb="popover"] [role="option"] *,
        [data-baseweb="menu"] li,
        [data-baseweb="menu"] li *,
        [role="listbox"] [role="option"],
        [role="listbox"] [role="option"] * {
            color: #111111 !important;
            -webkit-text-fill-color: #111111 !important;
            opacity: 1 !important;
        }
        /* Highlighted / hovered / currently-selected option: light-blue row so
           the dark text stays readable (the default was a near-white highlight
           that hid the selected item). */
        [data-baseweb="popover"] li:hover,
        [data-baseweb="popover"] [role="option"]:hover,
        [data-baseweb="popover"] [role="option"][aria-selected="true"],
        [data-baseweb="menu"] li[aria-selected="true"],
        [role="listbox"] [role="option"][aria-selected="true"] {
            background-color: #cfe6fb !important;
        }

        /* ---- Preset buttons + other secondary buttons ----
           Secondary buttons keep a light fill in dark mode, so their labels
           must be dark to stay readable. This covers the four preset buttons
           (Symmetric baseline, Cambered baseline, Biomimetic default, Workshop
           challenge) as well as Clear history and the CSV download button.
           Primary buttons are excluded — their white-on-slate text is handled
           above. */
        .stButton > button:not([kind="primary"]),
        .stDownloadButton > button {
            background-color: #eef4fa !important;
        }
        .stButton > button:not([kind="primary"]),
        .stButton > button:not([kind="primary"]) *,
        .stDownloadButton > button,
        .stDownloadButton > button * {
            color: #000000 !important;
        }
        /* Hold the black label on hover too (the light-mode hover rule would
           otherwise recolor it). */
        .stButton > button:not([kind="primary"]):hover,
        .stButton > button:not([kind="primary"]):hover *,
        .stDownloadButton > button:hover,
        .stDownloadButton > button:hover * {
            color: #000000 !important;
            border-color: #7fc4ee !important;
        }

        /* JSON / code surfaces */
        [data-testid="stJson"], pre, code {
            background-color: #1c1f26 !important;
        }

        /* Alerts (info / success / warning) made readable on dark */
        [data-testid="stAlert"], .stAlert {
            background-color: rgba(255, 255, 255, 0.07) !important;
        }

        /* Expander */
        [data-testid="stExpander"] details {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
        }

        /* Dividers */
        hr { border-color: rgba(255, 255, 255, 0.15) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "latest_input_dict" not in st.session_state:
    st.session_state.latest_input_dict = None
if "latest_label" not in st.session_state:
    st.session_state.latest_label = None
if "latest_clip_status" not in st.session_state:
    # None = interior prediction; otherwise a string from evaluate_clipping().
    st.session_state.latest_clip_status = None
if "latest_raw_output" not in st.session_state:
    st.session_state.latest_raw_output = None
if "latest_baseline" not in st.session_state:
    # dict from run_single_prediction() for a symmetric wing at the same flow
    # condition, or None when the selected family is not biomimetic.
    st.session_state.latest_baseline = None
if "prediction_history" not in st.session_state:
    # Accumulates one record per successful run (inputs + prediction).
    st.session_state.prediction_history = []

# -----------------------------------------------------------------------------
# Explanations
# -----------------------------------------------------------------------------
if show_explanations:
    intro_col1, intro_col2, intro_col3 = st.columns(3)
    with intro_col1:
        st.markdown(
            '<div class="info-card"><b>Input</b><br><span class="small-note">Wing family, tubercle geometry, chord lengths, sweep, angle of attack, and airspeed.</span></div>',
            unsafe_allow_html=True,
        )
    with intro_col2:
        st.markdown(
            '<div class="info-card"><b>Output</b><br><span class="small-note">Predicted separation location along the normalized chord, x/c.</span></div>',
            unsafe_allow_html=True,
        )
    with intro_col3:
        st.markdown(
            '<div class="info-card"><b>Sustainability Lens</b><br><span class="small-note">Use the prediction to discuss how efficient designs can reduce wasted flight energy.</span></div>',
            unsafe_allow_html=True,
        )

    with st.expander("What this prototype can and cannot do"):
        st.markdown(
            """
            **This app can:**
            - accept parameterized wing and flow inputs;
            - predict `separation_x_over_c` using the trained model;
            - visualize where separation is predicted to occur;
            - support a classroom discussion about energy use and sustainability.

            **This app cannot:**
            - replace full CFD simulation;
            - replace wind-tunnel testing;
            - accept raw STL/STEP CAD uploads;
            - certify a real aircraft or drone design;
            - directly calculate true lift, drag, battery life, or emissions.
            """
        )

    with st.expander("Model and input diagnostics"):
        st.write("Current input dictionary:")
        st.json(input_dict)
        st.write("Expected model feature order:")
        st.write(get_required_feature_columns())

        st.write("Validation metric diagnostics:")
        try:
            from src.inference import get_metrics_diagnostics  # noqa: E402

            diag = get_metrics_diagnostics()
            st.json(diag)

            if diag["resolved_metrics"]:
                st.success(diag["explanation"])
            elif diag["source_file"] is None:
                st.warning(diag["explanation"])
                st.code(
                    'import json, pathlib\n'
                    '# Run this at the end of Notebook 2, after evaluating on the test split.\n'
                    'from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error\n'
                    'import numpy as np\n\n'
                    'y_pred = model.predict(X_test)\n'
                    'metrics = {\n'
                    '    "r2": float(r2_score(y_test, y_pred)),\n'
                    '    "mae": float(mean_absolute_error(y_test, y_pred)),\n'
                    '    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),\n'
                    '}\n'
                    'pathlib.Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))\n'
                    'print(metrics)',
                    language="python",
                )
            else:
                st.error(diag["explanation"])
        except ImportError:
            st.info(
                "This build of src/inference.py predates get_metrics_diagnostics(). "
                "Falling back to the R² check."
            )
            raw_r2 = get_model_r2()
            st.write({"get_model_r2_score() returned": raw_r2})
            if raw_r2 is None:
                st.warning(
                    "No metrics file was found and DEFAULT_MODEL_R2 is not set. "
                    'Add models/metrics.json (e.g. {"r2": 0.93}) or set DEFAULT_MODEL_R2 '
                    "in src/inference.py."
                )
        except Exception as e:
            st.error(f"Could not read metrics from src.inference: {e!r}")
            st.info(
                "This usually means the app is not importing the updated src/inference.py. "
                "Confirm the new inference.py replaced the file at <project>/src/inference.py, "
                "then fully restart Streamlit (editing it while running is often not enough)."
            )

# -----------------------------------------------------------------------------
# Prediction block
# -----------------------------------------------------------------------------
if show_prediction:
    st.subheader("Model Prediction")

    st.caption("A later separation point, closer to x/c = 1, generally indicates more attached flow in this simplified screening context.")

    if submitted:
        try:
            # Round once, here, so the metric, the plot annotation, the flow
            # interpretation, and the CSV export can never disagree. Two
            # decimals is the right precision for a screening estimate; 0.7421
            # implies a measurement accuracy this model does not have.
            # Prefer the unclipped output so clipping can be reported honestly.
            # The fallback path receives an already-clipped value, in which case
            # only the boundary heuristic in evaluate_clipping() can detect it.
            # Clamping happens before rounding: a raw 1.4 must not round to 1.40
            # and be presented as if the model meant it.
            result = run_single_prediction(input_dict)
            model_output = result["raw"]
            clip_status = result["clip_status"]
            prediction = result["prediction"]

            # Compare to baseline: same flow condition, plain symmetric wing.
            # Wrapped separately so a baseline failure never discards the
            # user's actual prediction.
            baseline = None
            if input_dict["airfoil_family"] == "biomimetic":
                try:
                    baseline = run_single_prediction(build_symmetric_baseline_inputs(input_dict))
                except Exception:
                    baseline = None
            st.session_state.latest_baseline = baseline

            # Label is derived from the rounded value so a prediction of 0.795
            # cannot display as "0.80" while carrying the sub-0.80 label.
            label = describe_prediction(prediction)
            st.session_state.latest_prediction = prediction
            st.session_state.latest_input_dict = dict(input_dict)
            st.session_state.latest_label = label
            st.session_state.latest_clip_status = clip_status
            st.session_state.latest_raw_output = model_output

            # Append this run to the session history (most recent stays latest).
            record = {
                "run": len(st.session_state.prediction_history) + 1,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "preset": st.session_state.active_preset,
                **dict(input_dict),
                "predicted_separation_x_over_c": prediction,
                "raw_model_output": round(model_output, 4),
                "clipped": clip_status is not None,
                "symmetric_baseline_x_over_c": baseline["prediction"] if baseline else None,
                "delta_vs_baseline": (
                    round(prediction - baseline["prediction"], 2) if baseline else None
                ),
                "flow_interpretation": label,
            }
            st.session_state.prediction_history.append(record)
        except Exception as e:
            st.session_state.latest_prediction = None
            st.session_state.latest_input_dict = None
            st.session_state.latest_label = None
            st.session_state.latest_clip_status = None
            st.session_state.latest_baseline = None
            st.error(f"Prediction failed: {e}")
            st.info(
                "This is usually a model-file synchronization issue. Check that the saved .joblib model is present in the models/ folder, "
                "that the model filename matches what src/inference.py expects, and that the app inputs match the training feature schema exactly."
            )

    if st.session_state.latest_prediction is not None:
        prediction = st.session_state.latest_prediction
        label = st.session_state.latest_label
        clip_status = st.session_state.latest_clip_status
        raw_output = st.session_state.latest_raw_output
        baseline = st.session_state.latest_baseline
        metrics = get_model_metrics_safe()

        # The middle metric column lets the user pick which validation metric to
        # view. We only offer metrics that were actually found, so the dropdown
        # never promises a value the model file doesn't provide.
        available_metric_keys = [key for key in METRIC_KEYS if key in metrics]

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Predicted separation_x_over_c", f"{prediction:.2f}")
        m_col1.caption("This is a screening estimate, not a measured value")

        with m_col2:
            if available_metric_keys:
                selected_metric = st.selectbox(
                    "Validation metric",
                    available_metric_keys,
                    format_func=lambda key: METRIC_LABELS[key],
                    key="selected_metric",
                    label_visibility="collapsed",
                )
                st.metric(
                    METRIC_LABELS[selected_metric],
                    format_metric_value(selected_metric, metrics[selected_metric]),
                )
                st.caption(METRIC_HELP[selected_metric])
            else:
                st.metric("Model R²", "N/A")
                st.caption("No validation metrics recorded (see below).")

        # Two decimals on x/c means one decimal here would be false precision.
        m_col3.metric("Separation location", f"{prediction * 100:.0f}% chord")

        # Plain-language restatement of the number, directly beneath it.
        st.write(
            f"This means the model predicts separation about {prediction * 100:.0f}% of the way "
            "from the leading edge to the trailing edge."
        )

        # Held-out validation metrics, so the number above is read with the
        # model's actual accuracy in view rather than as an exact value. The
        # note always lists all available metrics; the dropdown above only
        # changes which one is highlighted in the metric row.
        if metrics:
            st.caption(format_reliability_note(metrics))

            # Turn MAE into an explicit band around the prediction. This is a
            # typical-error range, not a statistical confidence interval.
            if "mae" in metrics:
                low = max(0.0, prediction - metrics["mae"])
                high = min(1.0, prediction + metrics["mae"])
                st.caption(
                    f"Typical-error band: x/c ≈ {low:.2f} – {high:.2f}. "
                    "This is the model's average error on held-out data, not a confidence interval."
                )
        else:
            st.caption(
                "**Model reliability:** no validation metrics recorded. Add `models/metrics.json` "
                '(e.g. `{"r2": 0.93, "mae": 0.028, "rmse": 0.041}`) so this prediction can be '
                "reported with its held-out accuracy."
            )

        # A clipped value is a boundary artifact. Say so before the flow
        # interpretation, so it is never read as a confident result.
        if clip_status is not None:
            st.warning(f"**Prediction clipped:** {clipping_message(clip_status, raw_output)}")
        elif raw_output is not None and prediction in (0.0, 1.0):
            # Interior value that merely *rounds* onto a bound (e.g. 0.997 -> 1.00).
            # Without this note it would be indistinguishable from a clipped result.
            st.caption(
                f"Displayed as {prediction:.2f} by rounding; the model output was "
                f"{raw_output:.3f}, inside the valid range. This is not a clipped value."
            )

        # A sentence-style interpretation reads better as a colored callout than
        # as a metric (which is meant for short numeric values). When the value
        # was clipped, the confident green/blue styling would be misleading, so
        # the interpretation is downgraded to a warning callout.
        if clip_status is not None:
            st.warning(f"**Flow interpretation (unreliable):** {label}")
        elif prediction >= 0.80:
            st.success(f"**Flow interpretation:** {label}")
        elif prediction >= 0.60:
            st.info(f"**Flow interpretation:** {label}")
        else:
            st.warning(f"**Flow interpretation:** {label}")

        fig = plot_airfoil_and_separation(
            separation_x_over_c=prediction,
            dark_mode=dark_mode,
            baseline_x_over_c=baseline["prediction"] if baseline else None,
        )
        st.pyplot(fig)
        plt.close(fig)

        # -------------------------------------------------------------------
        # Compare to Baseline (biomimetic selections only)
        # -------------------------------------------------------------------
        if baseline is not None:
            st.markdown("#### Compare to Baseline")
            st.caption(
                "The same angle of attack and airspeed, run again on a plain symmetric wing "
                "with no tubercles."
            )

            delta = round(prediction - baseline["prediction"], 2)
            b_col1, b_col2, b_col3 = st.columns(3)
            b_col1.metric("Biomimetic (selected)", f"{prediction:.2f}")
            b_col2.metric("Symmetric baseline", f"{baseline['prediction']:.2f}")
            b_col3.metric(
                "Difference in x/c",
                f"{delta:+.2f}",
                delta=f"{delta * 100:+.0f}% chord",
            )

            st.write(describe_baseline_delta(delta))

            if baseline["clip_status"] is not None:
                st.warning(
                    "**Baseline clipped:** "
                    f"{clipping_message(baseline['clip_status'], baseline['raw'])} "
                    "The comparison above is unreliable because the baseline value is a "
                    "boundary artifact."
                )

            st.caption(
                "Both numbers come from the same screening model, so this is a comparison of "
                "predictions, not experimental evidence that tubercles delay separation. "
                "Confirming a real difference requires CFD or wind-tunnel testing."
            )

    # Full run history (every prediction made this session), shown independently
    # of the latest-run panel so it persists even after a failed run.
    if st.session_state.prediction_history:
        history_df = pd.DataFrame(st.session_state.prediction_history)

        record_header_col, clear_col = st.columns([3, 1])
        with record_header_col:
            st.write(f"Prediction record — all runs this session ({len(history_df)})")
        with clear_col:
            if st.button("Clear history", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

        st.dataframe(history_df, use_container_width=True)

        st.download_button(
            "Download all prediction records (CSV)",
            data=history_df.to_csv(index=False).encode("utf-8"),
            file_name="wingcheck_predictions.csv",
            mime="text/csv",
        )
    elif st.session_state.latest_prediction is None:
        st.info("Adjust the inputs in the Input Bar above, then click **Run Prediction**.")

# -----------------------------------------------------------------------------
# Sustainability section
# -----------------------------------------------------------------------------
if show_sustainability:
    st.subheader("Sustainability Lens")
    st.markdown(
        """
        Aerodynamically efficient designs can help drones and small aircraft use less
        energy because less energy is wasted fighting separated, turbulent flow. This
        section is an **educational scenario calculator**, not a certified emissions model.
        Use it to explore how small efficiency improvements could scale across many flights.
        """
    )

    if st.session_state.latest_prediction is not None:
        sustain_label, sustain_text = sustainability_interpretation(st.session_state.latest_prediction)
        st.markdown(f"**Prediction-based design note:** {sustain_label}")
        st.write(sustain_text)
    else:
        st.info("Run a prediction first to connect this sustainability estimate to your selected wing design.")

    calc_col1, calc_col2, calc_col3 = st.columns(3)
    with calc_col1:
        energy_per_flight_wh = st.number_input(
            "Estimated energy per flight (Wh)",
            min_value=1.0,
            max_value=100000.0,
            value=100.0,
            step=10.0,
        )
    with calc_col2:
        number_of_flights = st.number_input(
            "Number of flights",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
        )
    with calc_col3:
        carbon_factor_kg_per_kwh = st.slider(
            "Carbon factor (kg CO₂/kWh)",
            min_value=0.0,
            max_value=2.0,
            value=0.40,
            step=0.01,
        )

    sustainability_df = build_sustainability_table(
        energy_per_flight_wh=energy_per_flight_wh,
        number_of_flights=int(number_of_flights),
        carbon_factor_kg_per_kwh=carbon_factor_kg_per_kwh,
    )
    st.dataframe(sustainability_df, use_container_width=True)
    st.caption(
        "Important: the efficiency percentages above are what-if assumptions for discussion. The current model predicts separation location only, not true drag, battery life, or emissions."
    )

# -----------------------------------------------------------------------------
# Footer note
# -----------------------------------------------------------------------------
st.caption(
    "Prototype note: this is a lightweight ML screening tool connected to a saved notebook-trained model artifact. "
    "It should be used for exploration and education, not as a replacement for CFD, wind-tunnel testing, or professional aerodynamic design."
)

# -----------------------------------------------------------------------------
# About section (toggled from the sidebar)
# -----------------------------------------------------------------------------
if show_about:
    st.markdown("---")

    st.header("About the Tool")
    st.write(
        "This toolkit is meant to be a screening tool to help students and enthusiasts "
        "test potential aerofoil designs before moving onto expensive and time consuming "
        "CFD and physical testing. It is an educational tool. The toolkit helps users "
        "explore how aerodynamic design choices can connect to energy efficiency and "
        "sustainable engineering. The software for this toolkit can be found in the "
        "Github repository."
    )

    st.header("About the Creator")
    st.write(
        "This toolkit was created by Madhav S Anoop, a rising senior at Round Rock High "
        "School in Austin, Texas. Since he was little, Madhav has been passionate about "
        "aerospace engineering, physics, and mathematics. His research explores biomimetic "
        "wing designs that may help delay flow separation and improve aerodynamic "
        "efficiency. He intends to continue his research and pursue his passion as an "
        "aerospace engineer."
    )
