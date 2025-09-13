"""
Example Calculations and Demonstrations
======================================

This module provides comprehensive examples demonstrating the usage
of the Option Pricing Model with various scenarios and configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from option_pricing import OptionPricing
import pandas as pd


def example_1_basic_calculation():
    """
    Example 1: Basic option pricing calculation
    """
    print("Example 1: Basic Option Pricing")
    print("=" * 40)
    
    # Parameters for a typical call option
    S = 100.0  # Current stock price
    K = 105.0  # Strike price
    T = 0.25   # Time to expiration (3 months)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2  # Volatility (20%)
    
    option = OptionPricing(S, K, T, r, sigma, 'call')
    
    print(f"Stock Price: ${S}")
    print(f"Strike Price: ${K}")
    print(f"Time to Expiration: {T} years")
    print(f"Risk-free Rate: {r*100}%")
    print(f"Volatility: {sigma*100}%")
    print()
    
    # Compare all models
    results = option.compare_models()
    print("Model Comparison:")
    print(results.to_string(index=False))
    
    # Black-Scholes Greeks
    bs_price, delta, gamma, theta, vega = option.black_scholes()
    print(f"\nBlack-Scholes Greeks:")
    print(f"Option Price: ${bs_price:.4f}")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Theta: {theta:.4f}")
    print(f"Vega: {vega:.4f}")


def example_2_itm_otm_atm():
    """
    Example 2: In-the-money, Out-of-the-money, and At-the-money options
    """
    print("\nExample 2: ITM, OTM, and ATM Options")
    print("=" * 50)
    
    base_params = {
        'S': 100.0,
        'T': 0.25,
        'r': 0.05,
        'sigma': 0.2
    }
    
    scenarios = [
        {'K': 90, 'type': 'call', 'name': 'ITM Call (S=100, K=90)'},
        {'K': 100, 'type': 'call', 'name': 'ATM Call (S=100, K=100)'},
        {'K': 110, 'type': 'call', 'name': 'OTM Call (S=100, K=110)'},
        {'K': 90, 'type': 'put', 'name': 'OTM Put (S=100, K=90)'},
        {'K': 100, 'type': 'put', 'name': 'ATM Put (S=100, K=100)'},
        {'K': 110, 'type': 'put', 'name': 'ITM Put (S=100, K=110)'},
    ]
    
    results_data = []
    
    for scenario in scenarios:
        option = OptionPricing(
            base_params['S'], scenario['K'], base_params['T'],
            base_params['r'], base_params['sigma'], scenario['type']
        )
        
        bs_price, delta, gamma, theta, vega = option.black_scholes()
        
        results_data.append({
            'Scenario': scenario['name'],
            'Option_Price': bs_price,
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega
        })
    
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False))


def example_3_time_decay():
    """
    Example 3: Time decay analysis
    """
    print("\nExample 3: Time Decay Analysis")
    print("=" * 40)
    
    # Base parameters
    S = 100.0
    K = 105.0
    r = 0.05
    sigma = 0.2
    
    # Different time to expiration values
    times = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
    
    call_prices = []
    put_prices = []
    
    for T in times:
        call_option = OptionPricing(S, K, T, r, sigma, 'call')
        put_option = OptionPricing(S, K, T, r, sigma, 'put')
        
        call_price, _, _, _, _ = call_option.black_scholes()
        put_price, _, _, _, _ = put_option.black_scholes()
        
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    # Create DataFrame
    time_decay_df = pd.DataFrame({
        'Time_to_Expiration': times,
        'Call_Price': call_prices,
        'Put_Price': put_prices
    })
    
    print("Time Decay Analysis:")
    print(time_decay_df.to_string(index=False))
    
    # Plot time decay
    plt.figure(figsize=(10, 6))
    plt.plot(times, call_prices, 'b-', label='Call Option', linewidth=2)
    plt.plot(times, put_prices, 'r-', label='Put Option', linewidth=2)
    plt.xlabel('Time to Expiration (years)')
    plt.ylabel('Option Price')
    plt.title('Time Decay: Option Price vs Time to Expiration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # Show time decreasing
    plt.show()


def example_4_volatility_impact():
    """
    Example 4: Volatility impact on option prices
    """
    print("\nExample 4: Volatility Impact Analysis")
    print("=" * 45)
    
    # Base parameters
    S = 100.0
    K = 105.0
    T = 0.25
    r = 0.05
    
    # Different volatility levels
    volatilities = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    
    call_prices = []
    put_prices = []
    
    for sigma in volatilities:
        call_option = OptionPricing(S, K, T, r, sigma, 'call')
        put_option = OptionPricing(S, K, T, r, sigma, 'put')
        
        call_price, _, _, _, _ = call_option.black_scholes()
        put_price, _, _, _, _ = put_option.black_scholes()
        
        call_prices.append(call_price)
        put_prices.append(put_price)
    
    # Create DataFrame
    vol_impact_df = pd.DataFrame({
        'Volatility': [f"{v*100:.0f}%" for v in volatilities],
        'Call_Price': call_prices,
        'Put_Price': put_prices
    })
    
    print("Volatility Impact Analysis:")
    print(vol_impact_df.to_string(index=False))
    
    # Plot volatility impact
    plt.figure(figsize=(10, 6))
    plt.plot(volatilities, call_prices, 'b-', label='Call Option', linewidth=2, marker='o')
    plt.plot(volatilities, put_prices, 'r-', label='Put Option', linewidth=2, marker='s')
    plt.xlabel('Volatility')
    plt.ylabel('Option Price')
    plt.title('Volatility Impact on Option Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def example_5_model_convergence():
    """
    Example 5: Model convergence analysis
    """
    print("\nExample 5: Model Convergence Analysis")
    print("=" * 45)
    
    # Base parameters
    S = 100.0
    K = 105.0
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    option = OptionPricing(S, K, T, r, sigma, 'call')
    
    # Black-Scholes reference price
    bs_price, _, _, _, _ = option.black_scholes()
    
    # Binomial convergence
    binomial_steps = [10, 25, 50, 100, 200, 500, 1000]
    binomial_prices = []
    binomial_errors = []
    
    print("Binomial Model Convergence:")
    print("Steps\tPrice\t\tError")
    print("-" * 35)
    
    for steps in binomial_steps:
        bin_price, _, _ = option.binomial(steps)
        error = abs(bin_price - bs_price)
        binomial_prices.append(bin_price)
        binomial_errors.append(error)
        print(f"{steps}\t${bin_price:.6f}\t${error:.6f}")
    
    # Monte Carlo convergence
    mc_simulations = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    mc_prices = []
    mc_errors = []
    mc_std_errors = []
    
    print(f"\nMonte Carlo Model Convergence:")
    print("Simulations\tPrice\t\tError\t\tStd Error")
    print("-" * 55)
    
    for sims in mc_simulations:
        mc_price, mc_std_error, _ = option.monte_carlo(sims)
        error = abs(mc_price - bs_price)
        mc_prices.append(mc_price)
        mc_errors.append(error)
        mc_std_errors.append(mc_std_error)
        print(f"{sims}\t\t${mc_price:.6f}\t${error:.6f}\t${mc_std_error:.6f}")
    
    # Plot convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Binomial convergence
    ax1.plot(binomial_steps, binomial_errors, 'b-o', linewidth=2)
    ax1.set_xlabel('Number of Steps')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title('Binomial Model Convergence')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Monte Carlo convergence
    ax2.plot(mc_simulations, mc_errors, 'r-s', linewidth=2)
    ax2.set_xlabel('Number of Simulations')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Monte Carlo Model Convergence')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def example_6_greeks_analysis():
    """
    Example 6: Greeks analysis across different scenarios
    """
    print("\nExample 6: Greeks Analysis")
    print("=" * 35)
    
    # Base parameters
    S = 100.0
    K = 105.0
    T = 0.25
    r = 0.05
    sigma = 0.2
    
    # Different stock prices
    stock_prices = [80, 90, 100, 110, 120]
    
    greeks_data = []
    
    for stock_price in stock_prices:
        option = OptionPricing(stock_price, K, T, r, sigma, 'call')
        price, delta, gamma, theta, vega = option.black_scholes()
        
        greeks_data.append({
            'Stock_Price': stock_price,
            'Option_Price': price,
            'Delta': delta,
            'Gamma': gamma,
            'Theta': theta,
            'Vega': vega
        })
    
    greeks_df = pd.DataFrame(greeks_data)
    print("Greeks Analysis:")
    print(greeks_df.to_string(index=False))
    
    # Plot Greeks
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Delta
    ax1.plot(stock_prices, greeks_df['Delta'], 'b-o', linewidth=2)
    ax1.set_xlabel('Stock Price')
    ax1.set_ylabel('Delta')
    ax1.set_title('Delta vs Stock Price')
    ax1.grid(True, alpha=0.3)
    
    # Gamma
    ax2.plot(stock_prices, greeks_df['Gamma'], 'g-s', linewidth=2)
    ax2.set_xlabel('Stock Price')
    ax2.set_ylabel('Gamma')
    ax2.set_title('Gamma vs Stock Price')
    ax2.grid(True, alpha=0.3)
    
    # Theta
    ax3.plot(stock_prices, greeks_df['Theta'], 'r-^', linewidth=2)
    ax3.set_xlabel('Stock Price')
    ax3.set_ylabel('Theta')
    ax3.set_title('Theta vs Stock Price')
    ax3.grid(True, alpha=0.3)
    
    # Vega
    ax4.plot(stock_prices, greeks_df['Vega'], 'm-d', linewidth=2)
    ax4.set_xlabel('Stock Price')
    ax4.set_ylabel('Vega')
    ax4.set_title('Vega vs Stock Price')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def run_all_examples():
    """
    Run all examples in sequence
    """
    print("Option Pricing Model - Comprehensive Examples")
    print("=" * 60)
    
    example_1_basic_calculation()
    example_2_itm_otm_atm()
    example_3_time_decay()
    example_4_volatility_impact()
    example_5_model_convergence()
    example_6_greeks_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
