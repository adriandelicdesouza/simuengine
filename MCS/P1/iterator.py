import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate random numbers from different distributions
normal_samples = np.random.normal(loc=0, scale=1, size=1000)  # Mean=0, StdDev=1
uniform_samples = np.random.uniform(low=0, high=1, size=1000)  # Between 0-1
lognormal_samples = np.random.lognormal(mean=0, sigma=0.5, size=1000)  # Stock prices

print(f"Normal mean: {normal_samples.mean():.4f}")
print(f"Uniform mean: {uniform_samples.mean():.4f}")

# Simulate stock price over 252 trading days
n_simulations = 1000
n_days = 252

# Initial stock price
S0 = 100

# Parameters
mu = 0.10  # Expected annual return (10%)
sigma = 0.20  # Volatility (20%)
dt = 1/252  # Time step (1 day)

# Store all simulation paths
paths = np.zeros((n_simulations, n_days))
paths[:, 0] = S0  # Starting price

# Run simulation
for sim in range(n_simulations):
    for day in range(1, n_days):
        # Geometric Brownian Motion formula
        dW = np.random.normal(0, np.sqrt(dt))
        paths[sim, day] = paths[sim, day-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

print(f"Final prices - Min: ${paths[:, -1].min():.2f}, Max: ${paths[:, -1].max():.2f}")
print(f"Final prices - Mean: ${paths[:, -1].mean():.2f}")

