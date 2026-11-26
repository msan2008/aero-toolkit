"""WingCheck home page: tells visitors which mode to pick."""

import streamlit as st

st.title("🛩️ WingCheck")
st.markdown(
    "A machine-learning model, trained on ~1,700 CFD simulations, that predicts "
    "**where airflow separates from a wing**. Pick a mode:"
)

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("🏝️ Wing Lab")
        st.caption("New to aerodynamics? Start here. · ~15 min game")
        st.markdown(
            "- Glide Link's rescue message **across the ocean**\n"
            "- Unlock upgrades, up to a whale-inspired wing\n"
            "- Learn **flow separation** and **biomimicry**"
        )
        st.page_link("wing_lab.py", label="Play Wing Lab", icon="▶️")

with right:
    with st.container(border=True):
        st.subheader("🛠️ Advanced Mode")
        st.caption("The full WingCheck screening toolkit.")
        st.markdown(
            "- Any supported wing, angle and airspeed\n"
            "- Baseline comparison, saved runs, downloads\n"
            "- Diagnostics and a sustainability calculator"
        )
        st.page_link("advanced_mode.py", label="Open Advanced Mode", icon="▶️")

st.info(
    "An educational screening tool: it predicts separation location (x/c) only, not lift, drag "
    "or emissions, and doesn't replace CFD or wind-tunnel testing.",
    icon="ℹ️",
)
st.caption("Switch modes anytime from the sidebar.")
