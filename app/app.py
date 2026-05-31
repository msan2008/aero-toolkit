"""Streamlit app for the Aero Toolkit project.

This file is meant to live at:
    app/app.py

The app loads a saved model from models/, collects aerodynamic inputs from the
sidebar, runs inference through src/inference.py, and displays both the single
prediction and a sensitivity plot.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
# app.py lives inside app/. The project root is one folder above this file.
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (  # noqa: E402
    describe_model_response,
    describe_prediction,
    get_model_status,
    get_required_feature_columns,
    list_available_models,
    load_model,
    local_delta_report,
    make_sensitivity_sweep,
    predict_from_dict,
    validate_input_dict,
)

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aero Toolkit",
    page_icon="A",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def plot_airfoil_and_separation(
    separation_x_over_c: float,
    airfoil_family: str,
    angle_of_attack: float,
    airspeed: float,
) -> plt.Figure:
    """Create a lightweight conceptual chord plot with the predicted location."""
    fig, ax = plt.subplots(figsize=(8, 2.8))

    ax.plot([0, 1], [0, 0], linewidth=4)
    ax.scatter([separation_x_over_c], [0], s=120, zorder=3)
    ax.axvline(separation_x_over_c, linestyle="--", linewidth=1.5)

    ax.text(
        separation_x_over_c,
        0.08,
        f"x/c = {separation_x_over_c:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.25, 0.30)
    ax.set_xlabel("Normalized chord location, x/c")
    ax.set_yticks([])
    ax.set_title(
        f"Predicted separation location | {airfoil_family.title()} | "
        f"AoA={angle_of_attack:.2f} degrees | Airspeed={airspeed:.2f}"
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_sensitivity_curve(
    sweep_df: pd.DataFrame,
    variable: str,
    current_value: float,
    current_prediction: float,
) -> plt.Figure:
    """Plot the model prediction as one input variable is swept."""
    fig, ax = plt.subplots(figsize=(8, 4.2))

    ax.plot(
        sweep_df["sweep_value"],
        sweep_df["predicted_separation_x_over_c"],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
    )
    ax.axvline(current_value, linestyle="--", linewidth=1.4)
    ax.scatter([current_value], [current_prediction], s=120, zorder=4)

    ax.set_xlabel(variable.replace("_", " ").title())
    ax.set_ylabel("Predicted separation_x_over_c")
    ax.set_ylim(0, 1)
    ax.set_title("Local model sensitivity sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def build_input_dict() -> Dict[str, object]:
    """Collect sidebar inputs and return the model input dictionary."""
    st.sidebar.header("Input Parameters")

    airfoil_family = st.sidebar.selectbox(
        "Airfoil Family",
        ["symmetric", "cambered", "biomimetic"],
        index=2,
        help="Choose the wing family represented by this simplified input row.",
    )

    if airfoil_family == "biomimetic":
        tubercle_shape_options = ["whale", "biomimetic_v1"]
        tubercle_disabled = False
        default_amplitude = 28.5
        default_wavelength = 49.6
    else:
        tubercle_shape_options = ["none"]
        tubercle_disabled = True
        default_amplitude = 0.0
        default_wavelength = 0.0

    tubercle_shape = st.sidebar.selectbox("Tubercle Shape", tubercle_shape_options)

    tubercle_amplitude = st.sidebar.number_input(
        "Tubercle Amplitude",
        min_value=0.0,
        max_value=100.0,
        value=float(default_amplitude),
        step=0.1,
        disabled=tubercle_disabled,
    )

    tubercle_wavelength = st.sidebar.number_input(
        "Tubercle Wavelength",
        min_value=0.0,
        max_value=200.0,
        value=float(default_wavelength),
        step=0.1,
        disabled=tubercle_disabled,
    )

    root_chord = st.sidebar.number_input(
        "Root Chord",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.05,
    )

    tip_chord = st.sidebar.number_input(
        "Tip Chord",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.05,
    )

    sweep_angle = st.sidebar.number_input(
        "Sweep Angle, degrees",
        min_value=0.0,
        max_value=80.0,
        value=0.0,
        step=0.5,
    )

    angle_of_attack = st.sidebar.number_input(
        "Angle of Attack, degrees",
        min_value=-10.0,
        max_value=30.0,
        value=18.0,
        step=0.25,
        help="Use a smaller step size so the app can test local sensitivity.",
    )

    airspeed = st.sidebar.number_input(
        "Airspeed",
        min_value=0.1,
        max_value=300.0,
        value=30.0,
        step=0.5,
    )

    return {
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
# App body
# -----------------------------------------------------------------------------
st.title("Aerodynamic Screening Toolkit")
st.write(
    "This Version 1 app uses a saved machine learning pipeline to estimate "
    "`separation_x_over_c` from simplified geometry and flow inputs. The output "
    "is a fast screening estimate, not a replacement for full CFD."
)

with st.expander("Current Version 1 scope", expanded=False):
    st.markdown(
        """
        **What this app does**
        - accepts simplified wing and flow inputs
        - predicts `separation_x_over_c`
        - shows a simple risk interpretation
        - shows a sensitivity curve so users can see how the model responds

        **What this app does not do yet**
        - perform a full CFD simulation
        - accept raw 3D CAD files
        - model complete aircraft dynamics
        - guarantee smooth one-degree changes for tree-based models
        """
    )

available_models = list_available_models()
if not available_models:
    st.error("No .joblib model files were found in the models/ folder.")
    st.stop()

# Prefer the default model selected by inference.py, but allow manual choice in the UI.
try:
    default_status = get_model_status()
    default_model_name = Path(default_status["selected_model_file"]).name
except Exception:
    default_model_name = available_models[0]

default_index = available_models.index(default_model_name) if default_model_name in available_models else 0

st.sidebar.header("Model Settings")
selected_model_filename = st.sidebar.selectbox(
    "Model file",
    available_models,
    index=default_index,
    help="Choose which saved .joblib model from the models folder should drive the app.",
)

if st.sidebar.button("Reload selected model"):
    load_model(force_reload=True, explicit_model_filename=selected_model_filename)
    st.sidebar.success("Model reloaded.")

input_dict = build_input_dict()

# Validate once so the diagnostics and prediction use the same cleaned values.
try:
    cleaned_input = validate_input_dict(input_dict)
except Exception as exc:
    st.error(f"Input validation failed: {exc}")
    st.stop()

st.subheader("Model Output")

try:
    prediction = predict_from_dict(
        cleaned_input,
        explicit_model_filename=selected_model_filename,
    )
    interpretation = describe_prediction(prediction)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted separation_x_over_c", f"{prediction:.4f}")
    metric_col2.metric("Interpretation", interpretation)
    metric_col3.metric("Selected model", selected_model_filename)

    st.pyplot(
        plot_airfoil_and_separation(
            separation_x_over_c=prediction,
            airfoil_family=str(cleaned_input["airfoil_family"]),
            angle_of_attack=float(cleaned_input["angle_of_attack"]),
            airspeed=float(cleaned_input["airspeed"]),
        )
    )

except Exception as exc:
    st.error("The model could not generate a prediction.")
    st.exception(exc)
    st.stop()

# -----------------------------------------------------------------------------
# Sensitivity section
# -----------------------------------------------------------------------------
st.subheader("Sensitivity Check")
st.write(
    "This section tests whether the saved model changes when one input variable "
    "is moved around the current value. Tree-based models can be piecewise flat, "
    "so a one-degree change may not always change the prediction."
)

sensitivity_col1, sensitivity_col2, sensitivity_col3 = st.columns(3)
with sensitivity_col1:
    sensitivity_variable = st.selectbox(
        "Variable to sweep",
        ["angle_of_attack", "airspeed", "tubercle_amplitude", "tubercle_wavelength", "sweep_angle"],
        index=0,
    )
with sensitivity_col2:
    sweep_radius = st.number_input(
        "Sweep radius around current value",
        min_value=1.0,
        max_value=50.0,
        value=6.0,
        step=0.5,
    )
with sensitivity_col3:
    sweep_step = st.number_input(
        "Sweep step size",
        min_value=0.1,
        max_value=5.0,
        value=0.25,
        step=0.1,
    )

current_sweep_value = float(cleaned_input[sensitivity_variable])
sweep_min = current_sweep_value - float(sweep_radius)
sweep_max = current_sweep_value + float(sweep_radius)

# Keep physically constrained variables nonnegative in the sweep.
if sensitivity_variable in ["airspeed", "tubercle_amplitude", "tubercle_wavelength", "sweep_angle"]:
    sweep_min = max(0.0, sweep_min)
if sensitivity_variable == "airspeed":
    sweep_min = max(0.1, sweep_min)

try:
    sweep_df = make_sensitivity_sweep(
        cleaned_input,
        variable=sensitivity_variable,
        min_value=sweep_min,
        max_value=sweep_max,
        step=float(sweep_step),
        explicit_model_filename=selected_model_filename,
    )

    st.pyplot(
        plot_sensitivity_curve(
            sweep_df=sweep_df,
            variable=sensitivity_variable,
            current_value=current_sweep_value,
            current_prediction=prediction,
        )
    )

    delta_report = local_delta_report(
        cleaned_input,
        variable=sensitivity_variable,
        delta=1.0,
        explicit_model_filename=selected_model_filename,
    )

    st.write(describe_model_response(delta_report))

    delta_table = pd.DataFrame(
        [
            {
                "case": f"{sensitivity_variable} - 1",
                sensitivity_variable: delta_report["lower_value"],
                "prediction": delta_report["lower_prediction"],
            },
            {
                "case": "current input",
                sensitivity_variable: delta_report["current_value"],
                "prediction": delta_report["current_prediction"],
            },
            {
                "case": f"{sensitivity_variable} + 1",
                sensitivity_variable: delta_report["upper_value"],
                "prediction": delta_report["upper_prediction"],
            },
        ]
    )
    st.dataframe(delta_table, use_container_width=True, hide_index=True)

except Exception as exc:
    st.warning("Sensitivity check could not be completed.")
    st.exception(exc)

# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------
with st.expander("Model and input diagnostics", expanded=False):
    try:
        status = get_model_status(explicit_model_filename=selected_model_filename)
        st.write("Selected model file:", status["selected_model_file"])
        st.write("Selected metadata file:", status["selected_metadata_file"])
        st.write("Model type:", status["model_type"])
        st.write("Final estimator type:", status["final_estimator_type"])
        st.write("Model feature columns:")
        st.json(status["model_feature_columns"])
        st.write("Full app input columns:")
        st.json(get_required_feature_columns())
        st.write("Current cleaned input dictionary:")
        st.json(cleaned_input)
    except Exception as exc:
        st.exception(exc)

st.caption(
    "Note: If the output does not change for a one-degree input difference, that "
    "does not necessarily mean the app is broken. For tree-based models, local "
    "flat regions are expected. More simulation data around the same parameter "
    "range or a smoother model family can improve fine-grained sensitivity."
)
