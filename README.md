# Aero Toolkit: Biomimetic Drone Wing Screening Tool

## Summary
This project uses simulation data and machine learning to predict flow separation risk for standard and whale-inspired drone wing designs.

## Project Overview
Biomimetic wings have a strong capability of successfully improving aerodynamic efficiency for drones and other aircraft by delaying flow separation. However, current CFD solutions, may are not able to quickly predict flow separation, especially in complicated designs with biomimetic features. This tool uses machine learning to quickly predict the exact point of separation in a wing design, allowing engineers, scientists, and drone enthusiasts to quickly find a design that could potentially improve their craft's aerodynamic efficiency. 

## Why This Matters
Currently, its very time consuming and sometimes expensive for students, hobbyists, engineering clubs, and drone designers to create a design and then test it either in a CFD program or a physical wind tunnel. And even then, the design might not perform well and thus they would have to start from scratch with a new design. This web application allows for these users to quickly test a design to see if it could potentially work - and if it seems that it might be a good design, then they can continue with CFD/wind tunnel testing. This machine learning model can make the design process for building a drone much faster and less stressful. 

## What the App Does
The user has many options to customize their simulation. The most basic one is the angle of attack of the wing and the airspeed. The user can choose between a symmetrical wing (such as the NACA0012), a cambered wing, and of course, a biomimetic wing. They are also able to choose between the different amplitudes and wavelengths for their tubercles. The model outputs the "separation_x_over_c" which is the location of where the air flow is expected to separate from the surface of the aerofoil as a percentage of the aerofoil's chord. 

## Current Model Status
The dataset size currently has ~1750 datapoints - CFD simulations conducted at various angles of attack and airspeeds with different configurations of biomimetic wings, symmetric wings, and cambered wings. The current best performing model is the "extra trees" model.

## Repository Structure
app: the app folder contains the python file and README.md file that consists of the UI that allows the user to use the web application to run the model quickly and easily.
src: this is the source folder. It contains the utils.py, train_model.py, pre_process.py, and inference.py files.
data: this contains the dataset used to train the model.
models: this contains the various models that the user can choose from when using the web applications. 
notebooks: these are the notebooks that were used to create the .joblib model files.
outreach: this contains a spreadsheet with a list of our partners and other files related to this project's outreach work.
workshop: this folder contains the workshop's slides, worksheet, and feedback form.

## How to Run the App Locally
In depth instructions can be found in Streamlit's guides.
1. Create a folder called "aero-toolkit" in your computer's "Documents" folder.
2. In your computer's local terminal, create a virtual environment within the aero-toolkit folder.
3. Load the app.py file into the aero-toolkit folder. Create "models" and "src" folders within the aero-toolkit folder. Load the inference.py file into the src folder.
4. Once you are ready, activate your virtual environment.
5. To run streamlit, enter "streamlit run app.py"
6. This should create a "localhost" version of the web application that runs locally and not through the internet.

## Streamlit Deployment
The web application is run through Streamlit and is deployed from app/app.py.

## Files Needed for App Integration
app/app.py
src/inference.py
models/themodelname
requirements.txt

## Workshop Materials
The workshops are meant to teach how to use the web application and interact with the model as well as collecting feedback. Through slides, worksheets, and a feedback form, I am able to teach students, scientists, engineers, etc. how to use this model for their own needs but also learn on how I can improve it further. Workshop resources can be found in the repository's workshop folder.

## Current Limitations
This is a prototype screening tool, currently in Phase 2. This is NOT meant to be a replacement for full CFD, wind tunnel testing, or professional aerodynamic design. This is only meant to help scientists, engineers, drone enthusiasts, etc. with screening different designs quickly.

## Future Work
Future work will include expanding the dataset, physically validating the model output, better model tuning, an improved user interface, and community testing.

## Author
This model was created by Madhav S Anoop in 2026.
Contact the author at madhav.anoop2027@gmail.com

