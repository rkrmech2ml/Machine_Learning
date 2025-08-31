import numpy as np
from data_generator import (
    generate_collocation_points,
    generate_boundary_points,
    generate_initial_condition
)

# Define the initial condition: u(x, 0) = sin(pi * x)
u0_fn = lambda x: np.sin(np.pi * x)

# Generate points
colloc = generate_collocation_points(5000)
bound = generate_boundary_points(1000)
init_X, init_u = generate_initial_condition(1000, u0_fn=u0_fn)

# Show shapes
print("Collocation points:", colloc.shape)      # Should be (5000, 2)
print("Boundary points:", bound.shape)          # Should be (2000, 2)
print("Initial condition X:", init_X.shape)     # Should be (1000, 2)
print("Initial condition u:", init_u.shape)     # Should be (1000, 1)

# Optional: print first few values
print("\nFirst 5 collocation points:\n", colloc[:5])
print("\nFirst 5 boundary points:\n", bound[:5])
print("\nFirst 5 initial u values:\n", init_u[:5])
