import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pinn_model import PINN
from data_generator import (
    generate_collocation_points,
    generate_boundary_points,
    generate_initial_condition
)

# ----- Settings -----
N_colloc = 10000
N_bound = 2000
N_init = 2000
learning_rate = 0.001
epochs = 10000

# Initial condition function u(x, 0) = sin(pi x)
u0_fn = lambda x: np.sin(np.pi * x)

# ----- Generate training data -----
X_colloc = tf.convert_to_tensor(generate_collocation_points(N_colloc), dtype=tf.float32)
X_bound = tf.convert_to_tensor(generate_boundary_points(N_bound), dtype=tf.float32)
X_init_np, u_init_np = generate_initial_condition(N_init, u0_fn=u0_fn)
X_init = tf.convert_to_tensor(X_init_np, dtype=tf.float32)
u_init = tf.convert_to_tensor(u_init_np, dtype=tf.float32)

# ----- Model -----
model = PINN()

# ----- Optimizer -----
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_history = []

# ----- Training Step -----
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        # Physics loss
        f = model.compute_pde_residual(X_colloc)
        loss_f = tf.reduce_mean(tf.square(f))

        # Boundary loss
        u_bound = model(X_bound)
        loss_b = tf.reduce_mean(tf.square(u_bound))  # u=0 at boundaries

        # Initial condition loss
        u_pred_init = model(X_init)
        loss_i = tf.reduce_mean(tf.square(u_pred_init - u_init))

        total_loss = loss_f + loss_b + loss_i

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss

# ----- Training Loop -----
for epoch in range(1, epochs + 1):
    loss = train_step()
    loss_history.append(loss.numpy())
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy():.5e}")

# ----- Plot Loss History -----
plt.plot(loss_history)
plt.yscale('log')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
