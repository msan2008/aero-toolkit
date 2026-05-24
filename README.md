# Aero Toolkit: Biomimetic Drone Wing Screening Tool

## Project Overview
Biomimetic wings have a strong capability of successfully improving aerodynamic efficiency for drones and other aircraft by delaying flow separation. However, current CFD solutions, may are not able to quickly predict flow separation, especially in complicated designs with biomimetic features. This tool uses machine learning to quickly predict the exact point of separation in a wing design, allowing engineers, scientists, and drone enthusiasts to quickly find a design that could potentially improve their craft's aerodynamic efficiency. 

## Why This Matters
Currently, its very time consuming and sometimes expensive for students, hobbyists, engineering clubs, and drone designers to create a design and then test it either in a CFD program or a physical wind tunnel. And even then, the design might not perform well and thus they would have to start from scratch with a new design. This web application allows for these users to quickly test a design to see if it could potentially work - and if it seems that it might be a good design, then they can continue with CFD/wind tunnel testing. This machine learning model can make the design process for building a drone much faster and less stressful. 

## What the App Does
List the user inputs and the model output, especially `separation_x_over_c`.
The user has many options to customize their simulation. The most basic one is the angle of attack of the wing and the airspeed. The user can choose between a symmetrical wing (such as the NACA0012), a cambered wing, and of course, a biomimetic wing. They are also able to choose between the different amplitudes and wavelengths for their tubercles. They can 

## Current Model Status
State dataset size, model type, saved model file, and validation status.

## Repository Structure
Explain what each folder contains.

## How to Run the App Locally
Give exact terminal commands.

## Streamlit Deployment
Explain that the app is deployed from `app/app.py`.

## Files Needed for App Integration
List `app/app.py`, `src/inference.py`, `models/notebook2_gradient_boosting.joblib`, and `requirements.txt`.

## Workshop Materials
Mention slides, worksheet, and feedback form.

## Limitations
Clearly say this is a prototype screening tool, not a replacement for full CFD, wind tunnel testing, or professional aerodynamic design.

## Future Work
List dataset expansion, physical validation, better model tuning, improved UI, and community testing.

## Author
Name, project context, and contact if appropriate.

