import numpy as np
import pandas as pd
from option_pricing import OptionPricing
from visualization import plot_surface

def generate_scenarios(base_S, base_sigma, base_r, base_T, n=5, pct_range=0.2):
    """
    Generate a grid of scenarios for spot, vol, rate, and time.
    pct_range: +/- percent change from base
    n: number of points per dimension
    Returns: list of dicts with scenario parameters
    """
    S_range = np.linspace(base_S * (1-pct_range), base_S * (1+pct_range), n)
    sigma_range = np.linspace(base_sigma * (1-pct_range), base_sigma * (1+pct_range), n)
    r_range = np.linspace(base_r * (1-pct_range), base_r * (1+pct_range), n)
    T_range = np.linspace(base_T * (1-pct_range), base_T * (1+pct_range), n)
    scenarios = []
    for S in S_range:
        for sigma in sigma_range:
            for r in r_range:
                for T in T_range:
                    scenarios.append({'S': S, 'sigma': sigma, 'r': r, 'T': T})
    return scenarios

def run_scenario_analysis(option_df, scenario_list):
    """
    For each scenario, compute price and Greeks for all options in option_df.
    Returns: DataFrame with scenario columns and results.
    """
    results = []
    for scenario in scenario_list:
        for _, row in option_df.iterrows():
            op = OptionPricing(
                S=scenario['S'],
                K=row['strike'],
                T=scenario['T'],
                r=scenario['r'],
                sigma=scenario['sigma'],
                option_type=row['type']
            )
            price, delta, gamma, theta, vega, rho, vomma, vanna = op.black_scholes()
            results.append({
                'scenario_S': scenario['S'],
                'scenario_sigma': scenario['sigma'],
                'scenario_r': scenario['r'],
                'scenario_T': scenario['T'],
                'strike': row['strike'],
                'type': row['type'],
                'price': price,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'vomma': vomma,
                'vanna': vanna
            })
    return pd.DataFrame(results)

def plot_scenario_surface(df, scenario_param, greek='price'):
    """
    Plot surface for a given scenario parameter (e.g., S, sigma, r, T) vs strike.
    """
    plot_surface(df, x='strike', y=scenario_param, z=greek, title=f"Scenario: {greek} vs {scenario_param}")
