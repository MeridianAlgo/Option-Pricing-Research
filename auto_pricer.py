"""
Auto pricer with config file support. Usage:
python auto_pricer.py <TICKER> [--no-report] [--config config.yaml]
"""
import sys
import os
import pandas as pd
from data import get_stock_price, get_option_chain, get_historical_prices, get_risk_free_rate, estimate_historical_volatility
from option_pricing import OptionPricing
from visualization import plot_surface, plot_greeks_surface, plot_implied_vol_surface, export_html_report, export_excel

# Config support
try:
    import yaml
except ImportError:
    yaml = None

def load_config(path="config.yaml"):
    if yaml and os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return None


def auto_price(ticker, report=True):
    print(f"Fetching data for {ticker}...")
    S = get_stock_price(ticker)
    chain = get_option_chain(ticker)
    hist = get_historical_prices(ticker)
    sigma = estimate_historical_volatility(hist)
    r = get_risk_free_rate()
    T_today = pd.Timestamp.today()

    results = []
    for expiry, opt_data in chain.items():
        expiry_date = pd.Timestamp(expiry)
        T = (expiry_date - T_today).days / 365.0
        for opt_type in ['calls', 'puts']:
            df = opt_data[opt_type]
            for _, row in df.iterrows():
                K = float(row['strike'])
                iv = row.get('impliedVolatility', None)
                sigma_use = float(iv) if iv and iv > 0 else sigma
                option_type = 'call' if opt_type == 'calls' else 'put'
                op = OptionPricing(S, K, max(T, 1e-6), r, sigma_use, option_type)
                # Unpack all Greeks
                price, delta, gamma, theta, vega, rho, vomma, vanna = op.black_scholes()
                results.append({
                    'expiry': expiry,
                    'type': option_type,
                    'strike': K,
                    'S': S,
                    'T': T,
                    'r': r,
                    'sigma': sigma_use,
                    'impliedVolatility': iv,
                    'model': 'Black-Scholes',
                    'price': price,
                    'delta': delta,
                    'gamma': gamma,
                    'theta': theta,
                    'vega': vega,
                    'rho': rho,
                    'vomma': vomma,
                    'vanna': vanna
                })
    df_results = pd.DataFrame(results)
    print(df_results.head(20))
    df_results.to_csv(f"{ticker}_option_prices.csv", index=False)
    print(f"Results saved to {ticker}_option_prices.csv")

    if report:
        # Price surface (strike vs T)
        plot_surface(df_results[df_results['type']=='call'], x='strike', y='T', z='price', title=f"{ticker} Call Price Surface")
        plot_greeks_surface(df_results[df_results['type']=='call'], greek='delta', x='strike', y='T', title=f"{ticker} Call Delta Surface")
        from visualization import plot_all_greeks_surfaces
        plot_all_greeks_surfaces(df_results[df_results['type']=='call'], x='strike', y='T')
        plot_all_greeks_surfaces(df_results[df_results['type']=='put'], x='strike', y='T')
        if 'impliedVolatility' in df_results.columns:
            plot_implied_vol_surface(df_results[df_results['type']=='call'], x='strike', y='T', z='impliedVolatility', title=f"{ticker} Implied Vol Surface")
        export_html_report(df_results, filename=f"{ticker}_option_report.html")
        export_excel(df_results, filename=f"{ticker}_option_report.xlsx")
        print(f"HTML and Excel reports generated.")
        # Scenario analysis
        from scenario import generate_scenarios, run_scenario_analysis, plot_scenario_surface
        print("Running scenario analysis...")
        base_S = S
        base_sigma = df_results['sigma'].mean()
        base_r = r
        base_T = df_results['T'].mean()
        scenarios = generate_scenarios(base_S, base_sigma, base_r, base_T, n=5, pct_range=0.2)
        scenario_df = run_scenario_analysis(df_results[df_results['type']=='call'].head(5), scenarios)  # limit for speed
        scenario_df.to_csv(f"{ticker}_scenario_analysis.csv", index=False)
        print(f"Scenario analysis saved to {ticker}_scenario_analysis.csv")
        plot_scenario_surface(scenario_df, scenario_param='scenario_S', greek='price')
        plot_scenario_surface(scenario_df, scenario_param='scenario_sigma', greek='delta')
        # P&L simulation for a sample option
        from pnl_sim import simulate_pnl, compute_var_cvar
        print("Simulating P&L distribution for a sample call option...")
        sample = df_results[df_results['type']=='call'].iloc[0]
        pnl = simulate_pnl(sample['S'], sample['strike'], sample['T'], sample['r'], sample['sigma'], 'call', horizon=1/12, n_paths=10000)
        from visualization import plot_pnl_distribution
        plot_pnl_distribution(pnl, title=f"P&L Distribution for {ticker} Call Option")
        var, cvar = compute_var_cvar(pnl, alpha=0.05)
        print(f"5% VaR: {var:.2f}, 5% CVaR: {cvar:.2f}")
        # Diagnostics (if market prices available)
        try:
            from diagnostics import compute_errors, flag_outliers, plot_error_surface, plot_residuals_vs_features, print_model_fit_stats
            if 'market_price' in df_results.columns:
                print("Running diagnostics...")
                df_err, rmse, mae = compute_errors(df_results, market_col='market_price', model_col='price')
                df_err = flag_outliers(df_err)
                plot_error_surface(df_err, x='strike', y='T', z='residual', title=f"{ticker} Error Surface")
                plot_residuals_vs_features(df_err, features=['strike', 'T'])
                print_model_fit_stats(rmse, mae, df_err['outlier'].sum(), len(df_err))
        except Exception as e:
            print(f"Diagnostics skipped: {e}")
        # Placeholder: scenario analysis and P&L distribution
        # from visualization import plot_pnl_distribution
        # plot_pnl_distribution(pnl_array)

if __name__ == "__main__":
    config = None
    ticker = None
    report = True
    if '--config' in sys.argv:
        idx = sys.argv.index('--config')
        config = load_config(sys.argv[idx+1])
    elif os.path.exists('config.yaml'):
        config = load_config('config.yaml')
    if config:
        ticker = config.get('ticker', None)
        report = config.get('report', {}).get('generate_html', True)
    if not ticker:
        if len(sys.argv) < 2:
            print("Usage: python auto_pricer.py <TICKER> [--no-report] [--config config.yaml]")
            sys.exit(1)
        ticker = sys.argv[1].upper()
        report = True if len(sys.argv) < 3 or sys.argv[2] != '--no-report' else False
    auto_price(ticker, report=report)
