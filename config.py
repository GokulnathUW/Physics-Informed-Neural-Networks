# config.py
"""
The control panel for our fluid dynamics simulation.
All the important physics and training parameters are set here.
"""

# --- Physics Constants ---
# It's like setting the rules of our little universe.
RHO = 1.0          # Density of the fluid.
MU = 1e-5          # Viscosity of the fluid.
P_TOP = 1.0        # Pressure at the top boundary.
U_TOP = 0.5        # Velocity at the top boundary.

# --- Simulation & Model Parameters ---
MESH_SIZE = 11     # The resolution of our simulation grid. More points = more detail!
EPOCHS = 2000      # How many times our model will learn from the data.
BATCH_SIZE = 256   # How many data points the model sees at once.