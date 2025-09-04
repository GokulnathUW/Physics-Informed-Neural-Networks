# main.py
"""
The main script to run our Navier-Stokes PINN simulation.
Let's get this fluid dynamics party started!
"""
# We need to make our custom loss function globally accessible for the model.
# It's a bit of a quirk in how TensorFlow handles custom objects.
import model as pinn_model
global model

# Now, let's import everything else we need.
import config
import data_generator
import trainer

def main():
    """The main function that orchestrates the entire simulation."""
    # First, let's get the grid where our simulation will happen.
    X_train = data_generator.get_data(config.MESH_SIZE)
    
    # Let's build our Physics-Informed Neural Network.
    # The 'global' keyword is important here for our custom loss function to find the model.
    global model
    model = pinn_model.PINN(
        u_top=config.U_TOP,
        p_top=config.P_TOP,
        rho=config.RHO,
        mu=config.MU
    )
    
    # Time to start the training process!
    trainer.train(model, X_train, config.EPOCHS, config.BATCH_SIZE)
    
    print("\nTraining complete! The model has now learned the secrets of fluid flow (hopefully).")

# This is the "on" switch for our entire script.
if __name__ == "__main__":
    main()