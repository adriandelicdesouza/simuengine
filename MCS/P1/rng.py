import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate random numbers from different distributions
normal_samples = np.random.normal(loc=0, scale=1, size=1000)  # Mean=0, StdDev=1
uniform_samples = np.random.uniform(low=0, high=1, size=1000)  # Between 0-1
lognormal_samples = np.random.lognormal(mean=0, sigma=0.5, size=1000)  # Stock prices

print(f"Normal mean: {normal_samples.mean():.4f}")
print(f"Uniform mean: {uniform_samples.mean():.4f}")