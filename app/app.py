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
    """
    Create a simple conceptual wing-section plot and mark the predicted
    separation location along the normalized chord.

    This is intentionally lightweight. It is not a CFD visualization.
    """
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

    # Simple leading/trailing edge labels for workshop readability.
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
    """
    Provide a cautious, educational sustainability interpretation.
    This does not claim direct carbon savings from the ML model.
    """
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
    """
    Build a scenario table for classroom discussion. The percentages are not
    produced by the ML model. They are what-if assumptions for comparing how
    small efficiency gains can scale across many flights.
    """
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

show_explanations = st.toggle(
    "Show explanations",
    value=True,
    help="Turn off to hide the descriptive panels and expanders for a cleaner view.",
)

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

# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------
st.sidebar.header("1. Wing and Flow Inputs")
st.sidebar.caption("Drag the sliders, then run the model to see how the predicted separation point responds.")

airfoil_family = st.sidebar.selectbox(
    "Airfoil Family",
    ["symmetric", "cambered", "biomimetic"],
    index=2,
    help="The broad wing type. Biomimetic wings use nature-inspired features such as tubercles.",
)

# Tubercle controls only apply to biomimetic wings. Rendering them conditionally
# (instead of disabling them) means leftover amplitude/wavelength values can never
# leak into the model when a non-biomimetic family is selected.
if airfoil_family == "biomimetic":
    tubercle_shape = st.sidebar.selectbox(
        "Tubercle Shape",
        ["whale", "biomimetic_v1"],
        index=0,
        help="The tubercle pattern used for biomimetic wings.",
    )
    tubercle_amplitude = st.sidebar.slider(
        "Tubercle Amplitude",
        min_value=26.247,
        max_value=32.734,
        value=26.247,
        step=0.1,
        help="How tall the tubercle bumps are. Larger values represent more pronounced biomimetic features.",
    )
    tubercle_wavelength = st.sidebar.slider(
        "Tubercle Wavelength",
        min_value=42.337,
        max_value=49.607,
        value=49.607,
        step=0.1,
        help="The spacing between tubercle peaks. This helps define the biomimetic pattern.",
    )
else:
    tubercle_shape = "none"
    tubercle_amplitude = 0.0
    tubercle_wavelength = 0.0
    st.sidebar.caption("Tubercle controls appear when the airfoil family is set to *biomimetic*.")

# Wing geometry is fixed for this screening configuration, so it is shown as a
# read-only summary rather than as adjustable inputs.
root_chord = 1.0
tip_chord = 1.0
sweep_angle = 0.0

st.sidebar.markdown("**Fixed Wing Geometry**")
st.sidebar.markdown(
    f"- Root Chord: **{root_chord:.0f}**\n"
    f"- Tip Chord: **{tip_chord:.0f}**\n"
    f"- Sweep Angle: **{sweep_angle:.0f}°**"
)

angle_of_attack = st.sidebar.slider(
    "Angle of Attack (degrees)",
    min_value=0.0,
    max_value=25.0,
    value=10.0,
    step=0.5,
    help="The angle between the incoming airflow and the wing chord line.",
)

airspeed = st.sidebar.selectbox(
    "Airspeed",
    [15, 30],
    index=1,
    help="The relative speed of airflow over the wing.",
)

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
# Diagnostics
# -----------------------------------------------------------------------------
if show_explanations:
    with st.expander("Model and input diagnostics"):
        st.write("Current input dictionary:")
        st.json(input_dict)
        st.write("Expected model feature order:")
        st.write(get_required_feature_columns())
        st.caption(
            "If prediction fails, first check that the model file in models/ has the same filename expected by src/inference.py."
        )

# -----------------------------------------------------------------------------
# Prediction block
# -----------------------------------------------------------------------------
st.subheader("2. Model Prediction")

run_col, note_col = st.columns([1, 3])
with run_col:
    run_clicked = st.button("Run Prediction", type="primary", use_container_width=True)
with note_col:
    st.caption("A later separation point, closer to x/c = 1, generally indicates more attached flow in this simplified screening context.")

if "latest_prediction" not in st.session_state:
    st.session_state.latest_prediction = None
if "latest_input_dict" not in st.session_state:
    st.session_state.latest_input_dict = None
if "latest_label" not in st.session_state:
    st.session_state.latest_label = None

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

    # Plot geometry is read from the saved prediction inputs, not the live
    # sliders, so changing a slider without re-running cannot desync the chart
    # from the displayed prediction value.
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
    st.info("Adjust the inputs in the sidebar, then click **Run Prediction**.")

# -----------------------------------------------------------------------------
# Sustainability section
# -----------------------------------------------------------------------------
st.subheader("3. Sustainability Lens")
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
        help="Example: a small drone flight might use tens to hundreds of watt-hours. Use a value appropriate for your scenario.",
    )
with calc_col2:
    number_of_flights = st.number_input(
        "Number of flights",
        min_value=1,
        max_value=100000,
        value=100,
        step=10,
        help="Use the number of flights in a season, school year, club project, or test campaign.",
    )
with calc_col3:
    carbon_factor_kg_per_kwh = st.slider(
        "Carbon factor (kg CO₂/kWh)",
        min_value=0.0,
        max_value=2.0,
        value=0.40,
        step=0.01,
        help="Classroom placeholder. Update this value if you know the local electricity emissions factor.",
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
