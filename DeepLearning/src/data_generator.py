import numpy as np

def generate_collocation_points(n_points, x_range=(0, 1), t_range=(0, 1)):
    x = np.random.uniform(x_range[0], x_range[1], (n_points, 1))
    #(n_points, 1)= creates array with n_points rows and 1 column
    t = np.random.uniform(t_range[0], t_range[1], (n_points, 1))
    return np.hstack((x, t))  # shape (n_points, 2)

def generate_boundary_points(n_points, t_range=(0, 1)):
    t = np.random.uniform(t_range[0], t_range[1], (n_points, 1))
    left = np.hstack((np.zeros_like(t), t))       # x = 0
    right = np.hstack((np.ones_like(t), t))       # x = 1
    return np.vstack((left, right))               # shape (2*n_points, 2)

def generate_initial_condition(n_points, x_range=(0, 1), u0_fn=None):
    x = np.random.uniform(x_range[0], x_range[1], (n_points, 1))
    t = np.zeros_like(x)
    u0 = u0_fn(x) if u0_fn is not None else np.zeros_like(x)
    return np.hstack((x, t)), u0  # shape (n_points, 2), (n_points, 1)
