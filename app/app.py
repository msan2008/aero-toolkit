import streamlit as st

st.title("Madhav Aerodynamic Screening Toolkit")

tubercle_shape = st.selectbox(
    "Tubercle Shape",
    ["rectangular", "sinusoidal_pointed", "sinusoidal_curvy"]
)

tubercle_amplitude = st.slider("Tubercle Amplitude", 0.0, 1.0, 0.1)
tubercle_wavelength = st.slider("Tubercle Wavelength", 0.0, 10.0, 1.0)

root_chord = st.number_input("Root Chord", value=1.0)
tip_chord = st.number_input("Tip Chord", value=0.5)
sweep_angle = st.slider("Sweep Angle", 0.0, 45.0, 5.0)

angle_of_attack = st.slider("Angle of Attack", -5.0, 20.0, 5.0)
airspeed = st.number_input("Airspeed", value=30.0)

if st.button("Run Prediction"):
    st.write("Placeholder prediction goes here")


