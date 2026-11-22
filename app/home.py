"""WingCheck home page: tells visitors which mode to pick."""

import streamlit as st

st.title("🛩️ WingCheck")
st.markdown(
    "WingCheck uses a machine-learning model trained on about 1,700 OpenFOAM CFD "
    "simulations to predict **where airflow separates from a wing**. "
    "Pick how you'd like to explore it:"
)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("🐋 Wing Lab")
        st.caption("New to aerodynamics? Start here. · About 10 minutes")
        st.markdown(
            "- Learn what **biomimicry** and **flow separation** mean\n"
            "- Tune a whale-inspired wing in one guided challenge\n"
            "- Get **3 attempts** to beat the starting design\n"
            "- Answer a final question and download your results"
        )
        st.page_link("wing_lab.py", label="Start the Wing Lab challenge", icon="▶️")

with right:
    with st.container(border=True):
        st.subheader("🛠️ Advanced Mode")
        st.caption("The full WingCheck screening toolkit.")
        st.markdown(
            "- Choose the airfoil family, angle of attack, and airspeed\n"
            "- Compare against a symmetric baseline\n"
            "- Save, compare, and download multiple runs\n"
            "- Model diagnostics and a sustainability what-if calculator"
        )
        st.page_link("advanced_mode.py", label="Open Advanced Mode", icon="▶️")

st.info(
    "**What WingCheck is and isn't:** a fast, educational screening tool. It predicts one "
    "quantity, the separation location along the chord (`separation_x_over_c`). It does not "
    "predict lift, drag, battery life, or emissions, and it is not a replacement for CFD, "
    "wind-tunnel testing, or professional aerodynamic design.",
    icon="ℹ️",
)
st.caption("You can switch modes at any time using the menu in the sidebar.")
