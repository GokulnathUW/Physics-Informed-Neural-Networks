# data_generator.py
"""
The blueprint maker for our simulation world.
This module creates the 2D grid where our fluid will live.
"""
import numpy as np
import tensorflow as tf

def get_data(mesh_size):
    """
    Generates a 2D mesh grid for our simulation.
    Think of it as drawing the graph paper for our physics problem.
    """
    # Create evenly spaced points for our x and y axes.
    x = np.linspace(0, 1, mesh_size)
    y = np.linspace(0, 1, mesh_size)
    
    # Create the grid from these points.
    X, Y = np.meshgrid(x, y)
    
    # We need to flatten the grid into a list of (x, y) coordinates for our model.
    x_flat = X.flatten()
    y_flat = Y.flatten()
    
    # Combine them into a single tensor.
    X_train = tf.convert_to_tensor(np.vstack((x_flat, y_flat)).T, dtype=tf.float32)
    
    print(f"Generated a {mesh_size}x{mesh_size} mesh grid with {len(X_train)} points.")
    return X_train