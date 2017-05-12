import numpy as np
import lpa
np.random.seed(0)


m = 2  # Number of features
n = 10  # Number of simulated time points

# Generate linearly increasing data set
X = np.tile(np.arange(n), [m,1]) + np.random.randn(m,n)

# Train model with 1 component
n_components = 1
model = lpa.LPA(n_components, verbose=False)
model.fit(X)

# Readout components
W = model.get_components()
print W
