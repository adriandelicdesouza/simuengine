import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

# SET SEED ONCE AT THE TOP FOR REPRODUCIBILITY
# This locks in all randomness for the entire script
np.random.seed()

@dataclass
class SimulationResults:
    """Container for simulation results"""
    paths: np.ndarray  # Shape: (n_simulations, n_timesteps)
    final_values: np.ndarray  # Shape: (n_simulations,)
    stats: Dict
    
    def percentile(self, p):
        """Get percentile of final values"""
        return np.percentile(self.final_values, p)
    
    def probability_below(self, threshold):
        """Probability of ending below threshold"""
        return (self.final_values < threshold).sum() / len(self.final_values)


class MonteCarloSimulator:
    """Core Monte Carlo simulation engine"""
    
    def __init__(self, n_simulations: int = 1000):
        """
        Initialize simulator
        
        Args:
            n_simulations: Number of Monte Carlo runs
        
        Note: Set np.random.seed() at the top of your script for reproducibility
        """
        self.n_simulations = n_simulations
    
    def simulate_paths(self, 
                      initial_value: float,
                      n_steps: int,
                      step_generator) -> SimulationResults:
        """
        Run Monte Carlo simulation
        
        Args:
            initial_value: Starting value for each simulation
            n_steps: Number of time steps
            step_generator: Function that takes (current_value, step) -> next_value
        
        Returns:
            SimulationResults object with paths and statistics
        """
        # Initialize paths array
        paths = np.zeros((self.n_simulations, n_steps))
        paths[:, 0] = initial_value
        
        # Run simulations
        for step in range(1, n_steps):
            paths[:, step] = step_generator(paths[:, step-1], step)
        
        # Extract final values
        final_values = paths[:, -1]
        
        # Calculate statistics
        stats = self._calculate_stats(final_values)
        
        return SimulationResults(
            paths=paths,
            final_values=final_values,
            stats=stats
        )
    
    def _calculate_stats(self, values: np.ndarray) -> Dict:
        """Calculate comprehensive statistics"""
        return {
            'mean': float(values.mean()),
            'std': float(values.std()),
            'min': float(values.min()),
            'max': float(values.max()),
            'median': float(np.median(values)),
            'percentile_5': float(np.percentile(values, 5)),
            'percentile_25': float(np.percentile(values, 25)),
            'percentile_75': float(np.percentile(values, 75)),
            'percentile_95': float(np.percentile(values, 95)),
        }
    
    def print_results(self, results: SimulationResults):
        """Pretty print results"""
        print("\n" + "="*50)
        print("MONTE CARLO SIMULATION RESULTS")
        print("="*50)
        for key, value in results.stats.items():
            print(f"{key:20s}: {value:>10.2f}")
        print("="*50 + "\n")


# ============================================================================
# EXAMPLE 1: Simple Stock Price Simulation
# ============================================================================

def stock_price_step(S_prev, params):
    """
    Geometric Brownian Motion step
    Formula: S_t = S_{t-1} * exp((μ - σ²/2)dt + σ√dt * Z)
    where Z ~ N(0,1)
    """
    mu = params['mu']
    sigma = params['sigma']
    dt = params['dt']
    
    dW = np.random.normal(0, np.sqrt(dt), size=len(S_prev))
    return S_prev * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)


# Usage
simulator = MonteCarloSimulator(n_simulations=10000)

# Parameters
params = {
    'mu': 0.10,      # 10% annual return
    'sigma': 0.20,   # 20% volatility
    'dt': 1/252      # Daily steps
}

# Create step function with parameters bound
def step_fn(S_prev, step):
    return stock_price_step(S_prev, params)

# Run simulation: $100 initial, 252 trading days
results = simulator.simulate_paths(
    initial_value=100,
    n_steps=252,
    step_generator=step_fn
)

simulator.print_results(results)

# Extra insights
print(f"Probability stock doubles: {results.probability_below(200).__sub__(1):.2%}")
print(f"Probability stock loses 20%: {results.probability_below(80):.2%}")


# ============================================================================
# EXAMPLE 2: Portfolio Returns
# ============================================================================

class PortfolioSimulator(MonteCarloSimulator):
    """Extended simulator for portfolio returns"""
    
    def simulate_portfolio(self, 
                          initial_investment: float,
                          asset_returns: List[float],
                          asset_weights: List[float],
                          n_years: int = 10) -> SimulationResults:
        """
        Simulate portfolio growth
        
        Args:
            initial_investment: Starting amount
            asset_returns: Expected annual returns for each asset
            asset_weights: Portfolio weights (should sum to 1)
            n_years: Years to simulate
        """
        n_steps = n_years
        
        def step_fn(values, step):
            # Weighted return across assets
            weighted_return = sum(r*w for r,w in zip(asset_returns, asset_weights))
            # Add randomness (simplified - could be more sophisticated)
            noise = np.random.normal(weighted_return, 0.15, size=len(values))
            return values * (1 + noise)
        
        return self.simulate_paths(
            initial_value=initial_investment,
            n_steps=n_steps,
            step_generator=step_fn
        )


# Example: $100k portfolio, 70% stocks, 30% bonds
portfolio_sim = PortfolioSimulator(n_simulations=5000)
portfolio_results = portfolio_sim.simulate_portfolio(
    initial_investment=100000,
    asset_returns=[0.10, 0.04],  # Stocks: 10%, Bonds: 4%
    asset_weights=[0.70, 0.30],   # 70/30 split
    n_years=10
)

portfolio_sim.print_results(portfolio_results)
print(f"After 10 years, likely to have: ${portfolio_results.stats['median']:,.0f}")
print(f"Best 5% outcome: ${portfolio_results.stats['percentile_95']:,.0f}")
print(f"Worst 5% outcome: ${portfolio_results.stats['percentile_5']:,.0f}")