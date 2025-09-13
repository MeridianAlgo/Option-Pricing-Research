"""
Option Pricing Model Implementation
==================================

This module implements three different option pricing models:
1. Black-Scholes Model
2. Binomial Model  
3. Monte Carlo Model

Author: Option Pricing Model Project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from typing import Tuple, Optional
import pandas as pd


class OptionPricing:
    """
    A comprehensive class for option pricing using multiple models.
    """
    
    def __init__(self, S: float, K: float, T: float, r: float, sigma: float, 
                 option_type: str = 'call'):
        """
        Initialize option parameters.
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to expiration (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility of the underlying asset
        option_type : str
            'call' or 'put'
        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        
        if self.option_type not in ['call', 'put']:
            raise ValueError("option_type must be 'call' or 'put'")
    
    def black_scholes(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Calculate option price using Black-Scholes model.
        
        Returns:
        --------
        Tuple containing (option_price, delta, gamma, theta, vega, rho, vomma, vanna)
        """
        # Handle edge cases
        if self.T <= 0:
            # At expiration
            if self.option_type == 'call':
                option_price = max(self.S - self.K, 0)
                delta = 1.0 if self.S > self.K else 0.0
                rho = vomma = vanna = 0.0
            else:  # put
                option_price = max(self.K - self.S, 0)
                delta = -1.0 if self.S < self.K else 0.0
                rho = vomma = vanna = 0.0
            return option_price, delta, 0.0, 0.0, 0.0, rho, vomma, vanna
        
        if self.sigma <= 0:
            # Zero volatility case
            forward_price = self.S * np.exp(self.r * self.T)
            if self.option_type == 'call':
                option_price = max(forward_price - self.K, 0) * np.exp(-self.r * self.T)
                delta = 1.0 if forward_price > self.K else 0.0
                rho = vomma = vanna = 0.0
            else:  # put
                option_price = max(self.K - forward_price, 0) * np.exp(-self.r * self.T)
                delta = -1.0 if forward_price < self.K else 0.0
                rho = vomma = vanna = 0.0
            return option_price, delta, 0.0, 0.0, 0.0, rho, vomma, vanna
        
        # Calculate d1 and d2
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        # Calculate option price
        if self.option_type == 'call':
            option_price = self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:  # put
            option_price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            rho = -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-d2)
        
        # Calculate Greeks
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        
        if self.option_type == 'call':
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
        else:  # put
            theta = (-self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) 
                    + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-d2))
        
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        vomma = vega * d1 * d2 / self.sigma if self.sigma > 0 else 0.0
        vanna = vega * d2 / self.S if self.S > 0 else 0.0
        
        return option_price, delta, gamma, theta, vega, rho, vomma, vanna
    
    def binomial(self, n_steps: int = 100) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Calculate option price using Binomial model.
        
        Parameters:
        -----------
        n_steps : int
            Number of time steps in the binomial tree
            
        Returns:
        --------
        Tuple containing (option_price, stock_prices, option_prices)
        """
        dt = self.T / n_steps
        u = np.exp(self.sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor
        p = (np.exp(self.r * dt) - d) / (u - d)  # Risk-neutral probability
        
        # Initialize arrays for stock prices and option values
        stock_prices = np.zeros((n_steps + 1, n_steps + 1))
        option_values = np.zeros((n_steps + 1, n_steps + 1))
        
        # Calculate stock prices at each node
        for i in range(n_steps + 1):
            for j in range(i + 1):
                stock_prices[j, i] = self.S * (u ** (i - j)) * (d ** j)
        
        # Calculate option values at expiration
        for j in range(n_steps + 1):
            if self.option_type == 'call':
                option_values[j, n_steps] = max(0, stock_prices[j, n_steps] - self.K)
            else:  # put
                option_values[j, n_steps] = max(0, self.K - stock_prices[j, n_steps])
        
        # Backward induction to calculate option price
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j, i] = np.exp(-self.r * dt) * (p * option_values[j, i + 1] + 
                                                             (1 - p) * option_values[j + 1, i + 1])
        
        return option_values[0, 0], stock_prices, option_values
    
    def monte_carlo(self, n_simulations: int = 100000) -> Tuple[float, float, np.ndarray]:
        """
        Calculate option price using Monte Carlo simulation.
        
        Parameters:
        -----------
        n_simulations : int
            Number of Monte Carlo simulations
            
        Returns:
        --------
        Tuple containing (option_price, standard_error, simulated_payoffs)
        """
        np.random.seed(42)  # For reproducible results
        
        # Generate random stock price paths
        dt = self.T
        random_shocks = np.random.normal(0, 1, n_simulations)
        
        # Calculate stock prices at expiration using geometric Brownian motion
        stock_prices_T = self.S * np.exp((self.r - 0.5 * self.sigma**2) * dt + 
                                        self.sigma * np.sqrt(dt) * random_shocks)
        
        # Calculate payoffs
        if self.option_type == 'call':
            payoffs = np.maximum(stock_prices_T - self.K, 0)
        else:  # put
            payoffs = np.maximum(self.K - stock_prices_T, 0)
        
        # Discount payoffs to present value
        discounted_payoffs = payoffs * np.exp(-self.r * self.T)
        
        # Calculate option price and standard error
        option_price = np.mean(discounted_payoffs)
        standard_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        return option_price, standard_error, discounted_payoffs
    
    def heston(self, v0: float = None, kappa: float = None, theta: float = None, rho: float = None, sigma_v: float = None):
        """
        Placeholder for Heston stochastic volatility model.
        Parameters: v0, kappa, theta, rho, sigma_v (model params)
        Returns: NotImplementedError
        """
        raise NotImplementedError("Heston model not yet implemented.")

    def merton_jump_diffusion(self, lam: float = None, mu_j: float = None, sigma_j: float = None):
        """
        Placeholder for Merton jump diffusion model.
        Parameters: lam, mu_j, sigma_j (jump intensity, mean, std)
        Returns: NotImplementedError
        """
        raise NotImplementedError("Merton jump diffusion model not yet implemented.")
    
    def compare_models(self, n_steps: int = 100, n_simulations: int = 100000) -> pd.DataFrame:
        """
        Compare all three pricing models.
        
        Returns:
        --------
        pandas.DataFrame with results from all models
        """
        # Black-Scholes
        bs_price, delta, gamma, theta, vega, rho, vomma, vanna = self.black_scholes()
        
        # Binomial
        bin_price, _, _ = self.binomial(n_steps)
        
        # Monte Carlo
        mc_price, mc_error, _ = self.monte_carlo(n_simulations)
        
        results = pd.DataFrame({
            'Model': ['Black-Scholes', 'Binomial', 'Monte Carlo'],
            'Option_Price': [bs_price, bin_price, mc_price],
            'Standard_Error': [0, 0, mc_error],
            'Delta': [delta, 'N/A', 'N/A'],
            'Gamma': [gamma, 'N/A', 'N/A'],
            'Theta': [theta, 'N/A', 'N/A'],
            'Vega': [vega, 'N/A', 'N/A'],
            'Rho': [rho, 'N/A', 'N/A'],
            'Vomma': [vomma, 'N/A', 'N/A'],
            'Vanna': [vanna, 'N/A', 'N/A']
        })
        
        return results
    
    def plot_price_sensitivity(self, parameter: str, values: np.ndarray, 
                              model: str = 'black_scholes') -> None:
        """
        Plot option price sensitivity to a parameter.
        
        Parameters:
        -----------
        parameter : str
            Parameter to vary ('S', 'K', 'T', 'r', 'sigma')
        values : np.ndarray
            Array of parameter values
        model : str
            Pricing model to use ('black_scholes', 'binomial', 'monte_carlo')
        """
        prices = []
        
        for value in values:
            # Create temporary option with modified parameter
            temp_option = OptionPricing(
                S=self.S if parameter != 'S' else value,
                K=self.K if parameter != 'K' else value,
                T=self.T if parameter != 'T' else value,
                r=self.r if parameter != 'r' else value,
                sigma=self.sigma if parameter != 'sigma' else value,
                option_type=self.option_type
            )
            
            if model == 'black_scholes':
                price, _, _, _, _, _, _, _ = temp_option.black_scholes()
            elif model == 'binomial':
                price, _, _ = temp_option.binomial()
            elif model == 'monte_carlo':
                price, _, _ = temp_option.monte_carlo()
            else:
                raise ValueError("model must be 'black_scholes', 'binomial', or 'monte_carlo'")
            
            prices.append(price)
        
        plt.figure(figsize=(10, 6))
        plt.plot(values, prices, 'b-', linewidth=2, marker='o')
        plt.xlabel(f'{parameter}')
        plt.ylabel('Option Price')
        plt.title(f'Option Price Sensitivity to {parameter} ({self.option_type.title()} Option)')
        plt.grid(True, alpha=0.3)
        plt.show()


def main():
    """
    Example usage of the OptionPricing class.
    """
    # Example parameters
    S = 100.0  # Current stock price
    K = 105.0  # Strike price
    T = 0.25   # Time to expiration (3 months)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    # Create option pricing object
    option = OptionPricing(S, K, T, r, sigma, 'call')
    
    print("Option Pricing Model Comparison")
    print("=" * 50)
    print(f"Stock Price (S): ${S}")
    print(f"Strike Price (K): ${K}")
    print(f"Time to Expiration (T): {T} years")
    print(f"Risk-free Rate (r): {r*100}%")
    print(f"Volatility (σ): {sigma*100}%")
    print(f"Option Type: {option.option_type.title()}")
    print()
    
    # Compare all models
    results = option.compare_models()
    print("Model Comparison Results:")
    print(results.to_string(index=False))
    print()
    
    # Black-Scholes Greeks
    bs_price, delta, gamma, theta, vega, rho, vomma, vanna = option.black_scholes()
    print("Black-Scholes Greeks:")
    print(f"Option Price: ${bs_price:.4f}")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta: {theta:.4f}")
    print(f"Vega: {vega:.4f}")
    print(f"Rho: {rho:.4f}")
    print(f"Vomma: {vomma:.4f}")
    print(f"Vanna: {vanna:.4f}")
    print()
    
    # Sensitivity analysis
    print("Generating sensitivity plots...")
    
    # Stock price sensitivity
    stock_prices = np.linspace(80, 120, 20)
    option.plot_price_sensitivity('S', stock_prices)
    
    # Volatility sensitivity
    volatilities = np.linspace(0.1, 0.5, 20)
    option.plot_price_sensitivity('sigma', volatilities)


if __name__ == "__main__":
    main()
