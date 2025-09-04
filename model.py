# pinn_model.py
"""
The heart of the operation: The Physics-Informed Neural Network (PINN)! ❤️🧠
This file defines the model's architecture and the special loss function that teaches it physics.
"""
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

def custom_loss(x, y, u_top, p_top, rho, mu):
    """
    This is where the magic happens. This loss function doesn't just check predictions,
    it checks if the predictions obey the laws of physics (the Navier-Stokes equations).
    """
    # Use GradientTape to automatically track the derivatives we need. It's like a math detective!
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, y])
        
        # We need the gradient of u (x-velocity) with respect to our inputs.
        with tf.GradientTape(persistent=True) as u_tape:
            u_tape.watch([x, y])
            
            # Get the model's predictions for u, v, and p.
            X = tf.stack([x, y], axis=1)
            uvp = model(X)
            u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]

        # First derivatives of u.
        u_grads = u_tape.gradient(u, [x, y])
        u_x, u_y = u_grads[0], u_grads[1]

        # We need the gradient of v (y-velocity) as well.
        with tf.GradientTape(persistent=True) as v_tape:
            v_tape.watch([x, y])
            uvp = model(tf.stack([x, y], axis=1))
            u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]
            
        # First derivatives of v.
        v_grads = v_tape.gradient(v, [x, y])
        v_x, v_y = v_grads[0], v_grads[1]

    # Second derivatives (laplacians).
    u_xx = tape.gradient(u_x, x)
    u_yy = tape.gradient(u_y, y)
    v_xx = tape.gradient(v_x, x)
    v_yy = tape.gradient(v_y, y)
    
    # First derivatives of pressure (p).
    p_grads = tape.gradient(p, [x, y])
    p_x, p_y = p_grads[0], p_grads[1]

    del tape # Don't forget to clean up the tape!

    # --- The Physics Loss Equations ---
    # These are the Navier-Stokes equations that our model must learn to satisfy.
    
    # 1. Continuity Equation (Conservation of Mass)
    f1 = u_x + v_y
    
    # 2. Momentum Equation in x-direction
    f2 = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
    
    # 3. Momentum Equation in y-direction
    f3 = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
    
    # Calculate the mean squared error for our physics laws. We want this to be zero!
    loss1 = tf.reduce_mean(tf.square(f1))
    loss2 = tf.reduce_mean(tf.square(f2))
    loss3 = tf.reduce_mean(tf.square(f3))
    
    return loss1 + loss2 + loss3


class PINN(Model):
    """
    Our Physics-Informed Neural Network class.
    It's a standard neural network with a very special training routine.
    """
    def __init__(self, u_top, p_top, rho, mu, **kwargs):
        super().__init__(**kwargs)
        # Storing our physics constants.
        self.u_top = u_top
        self.p_top = p_top
        self.rho = rho
        self.mu = mu
        
        # Defining the metrics we want to watch during training.
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.l1_tracker = tf.keras.metrics.Mean(name="l1")
        self.l2_tracker = tf.keras.metrics.Mean(name="l2")

        # The neural network architecture itself. A few dense layers to learn the patterns.
        self.dense1 = Dense(50, activation='tanh')
        self.dense2 = Dense(50, activation='tanh')
        self.dense3 = Dense(50, activation='tanh')
        self.dense4 = Dense(50, activation='tanh')
        self.dense5 = Dense(3) # Output layer for u, v, and p.

    def call(self, inputs):
        """The forward pass of the network."""
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.dense5(x)

    @property
    def metrics(self):
        """Lets Keras know which metrics to track."""
        return [self.loss_tracker, self.l1_tracker, self.l2_tracker]

    def train_step(self, data):
        """
        This is the custom training step. It's where we calculate both the
        boundary condition loss and our special physics loss.
        """
        x, y = data # Unpack the data
        
        with tf.GradientTape() as tape:
            # --- Boundary Condition (BC) Loss ---
            # These are the known values at the edges of our simulation grid.
            
            # Find the points that lie on each boundary.
            top_wall = tf.cast(tf.where(y == 1)[:, 0], dtype=tf.int32)
            bottom_wall = tf.cast(tf.where(y == 0)[:, 0], dtype=tf.int32)
            left_wall = tf.cast(tf.where(x == 0)[:, 0], dtype=tf.int32)
            right_wall = tf.cast(tf.where(x == 1)[:, 0], dtype=tf.int32)
            
            # Get the model's predictions at these boundary points.
            uvp = self(tf.stack([x, y], axis=1))
            u_pred, v_pred, p_pred = uvp[:, 0], uvp[:, 1], uvp[:, 2]
            
            # Calculate the error between our predictions and the known boundary values.
            loss_t = tf.reduce_mean(tf.square(tf.gather(u_pred, top_wall) - self.u_top))
            loss_b = tf.reduce_mean(tf.square(tf.gather(u_pred, bottom_wall)))
            loss_l = tf.reduce_mean(tf.square(tf.gather(u_pred, left_wall)))
            loss_r = tf.reduce_mean(tf.square(tf.gather(u_pred, right_wall)))
            
            loss_vt = tf.reduce_mean(tf.square(tf.gather(v_pred, top_wall)))
            loss_vb = tf.reduce_mean(tf.square(tf.gather(v_pred, bottom_wall)))
            loss_vl = tf.reduce_mean(tf.square(tf.gather(v_pred, left_wall)))
            loss_vr = tf.reduce_mean(tf.square(tf.gather(v_pred, right_wall)))
            
            loss_p = tf.reduce_mean(tf.square(tf.gather(p_pred, top_wall) - self.p_top))
            
            bc_loss = loss_t + loss_b + loss_l + loss_r + loss_vt + loss_vb + loss_vl + loss_vr + loss_p
            
            # --- Physics Loss ---
            # Now, let's call our custom physics-based loss function.
            phy_loss = custom_loss(x, y, self.u_top, self.p_top, self.rho, self.mu)
            
            # The total loss is a combination of both.
            loss = bc_loss + phy_loss
            
        # Apply the gradients to update the model's weights.
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        # Update our metrics.
        self.loss_tracker.update_state(loss)
        self.l1_tracker.update_state(bc_loss)
        self.l2_tracker.update_state(phy_loss)
        
        return {"loss": self.loss_tracker.result(), "l1": self.l1_tracker.result(), "l2": self.l2_tracker.result()}