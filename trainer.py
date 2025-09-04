# trainer.py
"""
The training coach!
This module sets up and runs the training loop for our PINN model.
"""
import tensorflow as tf
from tqdm import tqdm # It's just not as fun without a progress bar!

def train(model, X_train, epochs, batch_size):
    """
    Handles the training process, including dataset preparation and the fit loop.
    """
    print("Preparing the dataset for training...")
    
    # We need to separate our x and y coordinates for the custom training step.
    x_train_coords = X_train[:, 0]
    y_train_coords = X_train[:, 1]
    
    # Let's batch our data to make training more efficient.
    dataset = tf.data.Dataset.from_tensor_slices((x_train_coords, y_train_coords)).batch(batch_size)
    
    # Time to compile the model with an optimizer. Adam is a great all-around choice.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    
    print(f"Starting training for {epochs} epochs...")
    # The main event! This is where the model learns.
    # We're using a simple progress bar here instead of Keras' default verbose output.
    for i in tqdm(range(epochs)):
        for x_batch, y_batch in dataset:
            metrics = model.train_on_batch(x_batch, y_batch)
        
        # Let's print the metrics every 100 epochs to check in on the progress.
        if i % 100 == 0:
            print(f"Epoch {i}: Loss: {metrics['loss']:.3e}, BC Loss: {metrics['l1']:.3f}, Physics Loss: {metrics['l2']:.5f}")