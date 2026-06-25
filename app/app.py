import sys
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

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aero Toolkit",
    page_icon="🛩️",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Lightweight styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp,
    .stMarkdown, .stText,
    h1, h2, h3, h4, h5, h6, p, span, label, div,
    button, input, select, textarea,
    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"] {
        font-family: 'Marcellus', 'Optima', 'Candara', serif !important;
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
        max-width: 1200px;
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
    
    /* Force the sidebar to be smaller */
    [data-testid="stSidebar"] {
        background-color: #eaf3fc;
        min-width: 220px !important;
        max-width: 220px !important;
    }
    
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
def plot_airfoil_and_separation(
    root_chord: float,
    tip_chord: float,
    sweep_angle: float,
    separation_x_over_c: float,
    airfoil_family: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))

    ax.plot([0, 1], [0, 0], linewidth=4)
    ax.scatter([separation_x_over_c], [0], s=120, zorder=3)
    ax.axvline(separation_x_over_c, linestyle="--", linewidth=1.5)
    ax.text(
        separation_x_over_c,
        0.08,
        f"x/c = {separation_x_over_c:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    ax.text(0.0, -0.11, "Leading edge", ha="left", va="top", fontsize=9)
    ax.text(1.0, -0.11, "Trailing edge", ha="right", va="top", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 0.35)
    ax.set_xlabel("Normalized chord location (x/c)")
    ax.set_yticks([])
    ax.set_title(
        f"Predicted Flow Separation Location | {airfoil_family.title()} Wing | "
        f"Root={root_chord:.2f}, Tip={tip_chord:.2f}, Sweep={sweep_angle:.1f}°"
    )
    ax.grid(True, axis="x", alpha=0.3)
    return fig


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
                "Estimated energy saved (kWh)": round(saved_kwh, 3),
                "Estimated CO₂ avoided (kg)": round(avoided_kg_co2, 3),
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
        "<h1 style='text-align: center; font-size: 3rem;'>Welcome to Aero-Toolkit</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;' class='small-note'>"
        "A lightweight machine-learning tool for screening biomimetic wing designs "
        "before committing to CFD or wind-tunnel testing."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 2vh;'></div>", unsafe_allow_html=True)

    spacer_left, center_col, spacer_right = st.columns([1, 2, 1])
    with center_col:
        if st.button(
            "Enter the Biomimetic Wing Screening Tool",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.entered = True
            st.rerun()
    st.stop()

# -----------------------------------------------------------------------------
# App title and overview
# -----------------------------------------------------------------------------
st.title("Aero Toolkit: Biomimetic Wing Screening Tool")
st.markdown(
    """
    This prototype uses a saved machine learning model to predict the flow separation
    location, `separation_x_over_c`, from wing geometry and flow inputs. The goal is
    to help students, makers, and robotics teams quickly screen wing ideas before
    committing to more expensive CFD or wind-tunnel testing.
    """
)

# -----------------------------------------------------------------------------
# Top Input Bar (Condensed Single Row)
# -----------------------------------------------------------------------------
st.markdown("---")
# Wrapper to apply tighter margins/labels via our custom CSS
st.markdown('<div class="condensed-label">', unsafe_allow_html=True)

root_chord = 1.0
tip_chord = 1.0
sweep_angle = 0.0

# 7 columns for a highly condensed, single-bar layout
cols = st.columns([1.2, 1, 1, 0.8, 1.2, 1, 1])

with cols[0]:
    airfoil_family = st.selectbox("Airfoil Family", ["symmetric", "cambered", "biomimetic"], index=2)
with cols[1]:
    angle_of_attack = st.slider("AoA (°)", min_value=0.0, max_value=25.0, value=10.0, step=0.5)
with cols[2]:
    airspeed = st.selectbox("Airspeed", [15, 30], index=1)
with cols[3]:
    st.markdown(
        "<div style='font-size:0.8rem; margin-top:0.4rem; opacity:0.8;'><b>Fixed Geo:</b><br>"
        f"Root: {root_chord}<br>Tip: {tip_chord}<br>Sweep: {sweep_angle}°</div>",
        unsafe_allow_html=True
    )

if airfoil_family == "biomimetic":
    with cols[4]:
        tubercle_shape = st.selectbox("Tubercle Shape", ["whale", "biomimetic_v1"], index=0)
    with cols[5]:
        tubercle_amplitude = st.slider("Amplitude", min_value=26.2, max_value=32.7, value=26.2, step=0.1)
    with cols[6]:
        tubercle_wavelength = st.slider("Wavelength", min_value=42.3, max_value=49.6, value=49.6, step=0.1)
else:
    tubercle_shape = "none"
    tubercle_amplitude = 0.0
    tubercle_wavelength = 0.0

st.markdown('</div>', unsafe_allow_html=True)
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
show_prediction = st.sidebar.toggle("Model Prediction", value=True)
show_sustainability = st.sidebar.toggle("Sustainability Lens", value=False)

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "latest_input_dict" not in st.session_state:
    st.session_state.latest_input_dict = None
if "latest_label" not in st.session_state:
    st.session_state.latest_label = None

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

# -----------------------------------------------------------------------------
# Prediction block
# -----------------------------------------------------------------------------
if show_prediction:
    st.subheader("Model Prediction")

    run_col, note_col = st.columns([1, 3])
    with run_col:
        run_clicked = st.button("Run Prediction", type="primary", use_container_width=True)
    with note_col:
        st.caption("A later separation point, closer to x/c = 1, generally indicates more attached flow in this simplified screening context.")

    if run_clicked:
        try:
            prediction = float(predict_from_dict(input_dict))
            label = describe_prediction(prediction)
            st.session_state.latest_prediction = prediction
            st.session_state.latest_input_dict = dict(input_dict)
            st.session_state.latest_label = label
        except Exception as e:
            st.session_state.latest_prediction = None
            st.session_state.latest_input_dict = None
            st.session_state.latest_label = None
            st.error(f"Prediction failed: {e}")
            st.info(
                "This is usually a model-file synchronization issue. Check that the saved .joblib model is present in the models/ folder, "
                "that the model filename matches what src/inference.py expects, and that the app inputs match the training feature schema exactly."
            )

    if st.session_state.latest_prediction is not None:
        prediction = st.session_state.latest_prediction
        label = st.session_state.latest_label
        saved = st.session_state.latest_input_dict

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Predicted separation_x_over_c", f"{prediction:.4f}")
        metric_col2.metric("Flow interpretation", label)
        metric_col3.metric("Separation location", f"{prediction * 100:.1f}% chord")

        fig = plot_airfoil_and_separation(
            root_chord=saved["root_chord"],
            tip_chord=saved["tip_chord"],
            sweep_angle=saved["sweep_angle"],
            separation_x_over_c=prediction,
            airfoil_family=saved["airfoil_family"],
        )
        st.pyplot(fig)
        plt.close(fig)

        output_df = pd.DataFrame([saved])
        output_df["predicted_separation_x_over_c"] = prediction
        st.write("Prediction record")
        st.dataframe(output_df, use_container_width=True)
    else:
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
        st.caption("Run a prediction first to connect the sustainability discussion to the selected wing design.")

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
