"""
Main Application for Option Pricing Model
========================================

Interactive command-line interface for the Option Pricing Model.
Allows users to input parameters and compare different pricing models.
"""

import sys
import numpy as np
from option_pricing import OptionPricing


def get_user_input():
    """
    Get option parameters from user input.
    """
    print("Option Pricing Model Calculator")
    print("=" * 40)
    print()
    
    try:
        S = float(input("Enter current stock price (S): $"))
        K = float(input("Enter strike price (K): $"))
        T = float(input("Enter time to expiration in years (T): "))
        r = float(input("Enter risk-free interest rate (r) as decimal (e.g., 0.05 for 5%): "))
        sigma = float(input("Enter volatility (σ) as decimal (e.g., 0.2 for 20%): "))
        
        print("\nOption type:")
        print("1. Call")
        print("2. Put")
        option_choice = input("Choose option type (1 or 2): ").strip()
        
        if option_choice == "1":
            option_type = "call"
        elif option_choice == "2":
            option_type = "put"
        else:
            print("Invalid choice. Defaulting to call option.")
            option_type = "call"
        
        return S, K, T, r, sigma, option_type
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return None


def display_results(option):
    """
    Display comprehensive results from all pricing models.
    """
    print("\n" + "=" * 60)
    print("OPTION PRICING RESULTS")
    print("=" * 60)
    
    # Display input parameters
    print(f"\nInput Parameters:")
    print(f"Stock Price (S): ${option.S:.2f}")
    print(f"Strike Price (K): ${option.K:.2f}")
    print(f"Time to Expiration (T): {option.T:.4f} years")
    print(f"Risk-free Rate (r): {option.r:.4f} ({option.r*100:.2f}%)")
    print(f"Volatility (σ): {option.sigma:.4f} ({option.sigma*100:.2f}%)")
    print(f"Option Type: {option.option_type.title()}")
    
    # Compare all models
    print(f"\nModel Comparison:")
    print("-" * 40)
    results = option.compare_models()
    
    for _, row in results.iterrows():
        print(f"{row['Model']:15}: ${row['Option_Price']:.4f}")
        if row['Standard_Error'] > 0:
            print(f"{'Standard Error':15}: ±${row['Standard_Error']:.4f}")
    
    # Black-Scholes Greeks
    print(f"\nBlack-Scholes Greeks:")
    print("-" * 40)
    bs_price, delta, gamma, theta, vega = option.black_scholes()
    print(f"Option Price: ${bs_price:.4f}")
    print(f"Delta:        {delta:.4f}")
    print(f"Gamma:        {gamma:.4f}")
    print(f"Theta:        {theta:.4f}")
    print(f"Vega:         {vega:.4f}")
    
    # Interpretation
    print(f"\nInterpretation:")
    print("-" * 40)
    print(f"• Delta ({delta:.4f}): Price sensitivity to stock price changes")
    print(f"• Gamma ({gamma:.4f}): Delta sensitivity to stock price changes")
    print(f"• Theta ({theta:.4f}): Price sensitivity to time decay")
    print(f"• Vega ({vega:.4f}): Price sensitivity to volatility changes")


def sensitivity_analysis_menu(option):
    """
    Menu for sensitivity analysis options.
    """
    while True:
        print(f"\nSensitivity Analysis Menu:")
        print("1. Stock Price Sensitivity")
        print("2. Strike Price Sensitivity")
        print("3. Time to Expiration Sensitivity")
        print("4. Risk-free Rate Sensitivity")
        print("5. Volatility Sensitivity")
        print("6. Return to Main Menu")
        
        choice = input("\nSelect analysis (1-6): ").strip()
        
        if choice == "1":
            min_price = float(input("Enter minimum stock price: $"))
            max_price = float(input("Enter maximum stock price: $"))
            prices = np.linspace(min_price, max_price, 20)
            option.plot_price_sensitivity('S', prices)
            
        elif choice == "2":
            min_strike = float(input("Enter minimum strike price: $"))
            max_strike = float(input("Enter maximum strike price: $"))
            strikes = np.linspace(min_strike, max_strike, 20)
            option.plot_price_sensitivity('K', strikes)
            
        elif choice == "3":
            min_time = float(input("Enter minimum time to expiration (years): "))
            max_time = float(input("Enter maximum time to expiration (years): "))
            times = np.linspace(min_time, max_time, 20)
            option.plot_price_sensitivity('T', times)
            
        elif choice == "4":
            min_rate = float(input("Enter minimum risk-free rate (decimal): "))
            max_rate = float(input("Enter maximum risk-free rate (decimal): "))
            rates = np.linspace(min_rate, max_rate, 20)
            option.plot_price_sensitivity('r', rates)
            
        elif choice == "5":
            min_vol = float(input("Enter minimum volatility (decimal): "))
            max_vol = float(input("Enter maximum volatility (decimal): "))
            vols = np.linspace(min_vol, max_vol, 20)
            option.plot_price_sensitivity('sigma', vols)
            
        elif choice == "6":
            break
        else:
            print("Invalid choice. Please try again.")


def advanced_options_menu(option):
    """
    Menu for advanced options and customizations.
    """
    while True:
        print(f"\nAdvanced Options Menu:")
        print("1. Custom Binomial Steps")
        print("2. Custom Monte Carlo Simulations")
        print("3. Model Accuracy Comparison")
        print("4. Return to Main Menu")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            try:
                steps = int(input("Enter number of binomial steps (default 100): ") or "100")
                bin_price, _, _ = option.binomial(steps)
                print(f"\nBinomial Model Price ({steps} steps): ${bin_price:.4f}")
            except ValueError:
                print("Invalid input.")
                
        elif choice == "2":
            try:
                sims = int(input("Enter number of Monte Carlo simulations (default 100000): ") or "100000")
                mc_price, mc_error, _ = option.monte_carlo(sims)
                print(f"\nMonte Carlo Price ({sims} simulations): ${mc_price:.4f}")
                print(f"Standard Error: ±${mc_error:.4f}")
            except ValueError:
                print("Invalid input.")
                
        elif choice == "3":
            print(f"\nModel Accuracy Comparison:")
            print("-" * 40)
            
            # Black-Scholes (analytical)
            bs_price, _, _, _, _ = option.black_scholes()
            
            # Binomial with different step sizes
            steps_list = [10, 50, 100, 500, 1000]
            print("Binomial Model Convergence:")
            for steps in steps_list:
                bin_price, _, _ = option.binomial(steps)
                error = abs(bin_price - bs_price)
                print(f"{steps:4d} steps: ${bin_price:.4f} (Error: ${error:.4f})")
            
            # Monte Carlo with different simulation counts
            sims_list = [1000, 10000, 100000, 1000000]
            print("\nMonte Carlo Model Convergence:")
            for sims in sims_list:
                mc_price, mc_error, _ = option.monte_carlo(sims)
                error = abs(mc_price - bs_price)
                print(f"{sims:7d} sims: ${mc_price:.4f} (Error: ${error:.4f}, StdErr: ±${mc_error:.4f})")
                
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")


def main_menu():
    """
    Main application menu.
    """
    while True:
        print(f"\n{'='*60}")
        print("OPTION PRICING MODEL - MAIN MENU")
        print(f"{'='*60}")
        print("1. Calculate Option Price")
        print("2. Sensitivity Analysis")
        print("3. Advanced Options")
        print("4. Example Calculations")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == "1":
            params = get_user_input()
            if params:
                S, K, T, r, sigma, option_type = params
                option = OptionPricing(S, K, T, r, sigma, option_type)
                display_results(option)
                
        elif choice == "2":
            params = get_user_input()
            if params:
                S, K, T, r, sigma, option_type = params
                option = OptionPricing(S, K, T, r, sigma, option_type)
                sensitivity_analysis_menu(option)
                
        elif choice == "3":
            params = get_user_input()
            if params:
                S, K, T, r, sigma, option_type = params
                option = OptionPricing(S, K, T, r, sigma, option_type)
                advanced_options_menu(option)
                
        elif choice == "4":
            show_examples()
            
        elif choice == "5":
            print("Thank you for using the Option Pricing Model!")
            break
            
        else:
            print("Invalid choice. Please try again.")


def show_examples():
    """
    Display example calculations with different scenarios.
    """
    print(f"\n{'='*60}")
    print("EXAMPLE CALCULATIONS")
    print(f"{'='*60}")
    
    examples = [
        {
            "name": "ITM Call Option",
            "S": 110, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.2, "type": "call"
        },
        {
            "name": "OTM Put Option", 
            "S": 90, "K": 100, "T": 0.5, "r": 0.03, "sigma": 0.25, "type": "put"
        },
        {
            "name": "ATM Call Option",
            "S": 100, "K": 100, "T": 0.1, "r": 0.02, "sigma": 0.15, "type": "call"
        },
        {
            "name": "High Volatility Put",
            "S": 95, "K": 105, "T": 1.0, "r": 0.04, "sigma": 0.4, "type": "put"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}: {example['name']}")
        print("-" * 40)
        
        option = OptionPricing(
            example['S'], example['K'], example['T'], 
            example['r'], example['sigma'], example['type']
        )
        
        results = option.compare_models()
        
        print(f"Parameters: S=${example['S']}, K=${example['K']}, T={example['T']}yr, r={example['r']*100}%, σ={example['sigma']*100}%")
        print("Model Results:")
        for _, row in results.iterrows():
            print(f"  {row['Model']:15}: ${row['Option_Price']:.4f}")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your inputs and try again.")
