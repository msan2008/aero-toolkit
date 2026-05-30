#Columns
Airfoil_family: This represents what type of aerofoil the design being tested is - biomimetic, symmetric, or cambered.
tubercle_amplitude: Since the biomimetic tubercles are in the shape of a sinusoidal wave, the "length" of the tubercles from the bottom to the top is the amplitude.
tubercle_wavelength: This is the wavelength of the sinusoidal shape the tubercle takes. This is the distance between the tops of each tubercle.
tubercle_shape: symmetric and cambered designs have no tubercles, so they are categorized as "none." For biomimetic ones, though, since the current designs only make up of whale mimetic designs, the tubercle shape is "whale."
root_chord: This is the chord length at the point where the wing connects to the aircraft's fuselage. In this iteration though, all the designs have the exact same chord length. Future iterations of the dataset will include variations in the root chord.
tip_chord: This is the chord length at the point farthest away from the aircraft's fuselage. In this iteration though, all the designs have the exact same chord length. Future iterations of the dataset will include variations in the root chord.
sweep_angle: This is the angle at which the wing is "tilted" in respect to the fuselage of the aircraft. In this dataset's iteration, though, all wing designs have no sweep angle. 
angle_of_attack: This is the angle at which the wing design faces incoming airflow. 
airspeed: The velocity of the air. 
separation_x_over_c: This is the predicted location at which airflow separates from the upper surface of the wing design as a percentage of its chord.
data_source: Details where the data came from - either OpenFOAM or Autodesk CFD. 
