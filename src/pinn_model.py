import tensorflow as tf

class PINN(tf.keras.Model):
    def __init__(self, hidden_layers=4, neurons_per_layer=20):
        super(PINN, self).__init__()
        self.hidden = [tf.keras.layers.Dense(neurons_per_layer, activation='tanh') for _ in range(hidden_layers)]
        self.output_layer = tf.keras.layers.Dense(1)  # Output: u(x, t)

    def call(self, inputs):
        x, t = inputs[:, 0:1], inputs[:, 1:2]
        X = tf.concat([x, t], axis=1)
        z = X
        for layer in self.hidden:
            z = layer(z)
        u = self.output_layer(z)
        return u

    def compute_pde_residual(self, inputs):
        """Computes the residual of the 1D heat equation: u_t = u_xx"""
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(inputs)
            u = self.call(inputs)
            u_x_t = tape1.gradient(u, inputs)  # ∂u/∂x and ∂u/∂t

        u_x = u_x_t[:, 0:1]
        u_t = u_x_t[:, 1:2]

        with tf.GradientTape() as tape2:
            tape2.watch(inputs)
            u_xx_t = tape2.gradient(u_x, inputs)

        u_xx = u_xx_t[:, 0:1]

        residual = u_t - u_xx
        return residual
