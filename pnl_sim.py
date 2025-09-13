import numpy as np
from option_pricing import OptionPricing
from visualization import plot_pnl_distribution

def simulate_pnl(S0, K, T, r, sigma, option_type, horizon=1/12, n_paths=10000):
    """
    Simulate P&L for an option over a horizon (years) using Monte Carlo.
    Returns: array of P&L values
    """
    # Simulate spot at horizon
    S1 = S0 * np.exp((r - 0.5 * sigma**2) * horizon + sigma * np.sqrt(horizon) * np.random.randn(n_paths))
    # Price at t=0
    op0 = OptionPricing(S0, K, T, r, sigma, option_type)
    price0, *_ = op0.black_scholes()
    # Price at t=horizon (T decreases)
    op1 = OptionPricing(S1, K, max(T-horizon, 1e-6), r, sigma, option_type)
    price1 = np.array([op1.black_scholes()[0] for S1 in S1])
    # P&L: change in option value
    pnl = price1 - price0
    return pnl

def compute_var_cvar(pnl, alpha=0.05):
    """
    Compute Value at Risk (VaR) and Conditional VaR (CVaR) at level alpha.
    """
    var = np.percentile(pnl, 100*alpha)
    cvar = pnl[pnl <= var].mean()
    return var, cvar

# plot_pnl_distribution is already in visualization.py
