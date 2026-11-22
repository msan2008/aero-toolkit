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
        st.subheader("🏝️ Wing Lab")
        st.caption("New to aerodynamics? Start here. · A game, about 15 minutes")
        st.markdown(
            "- Help Link send a rescue message in a bottle **across the ocean**\n"
            "- Launch three gliders per round and watch them fly\n"
            "- Earn points to **unlock upgrades**, up to a whale-inspired wing\n"
            "- Learn about **biomimicry** and **flow separation** along the way"
        )
        st.page_link("wing_lab.py", label="Play Wing Lab", icon="▶️")

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
