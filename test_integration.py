import unittest
import numpy as np
import pandas as pd
import torch
from option_pricing import OptionPricing
from ml_model import fit_ml_model, fit_lstm_model, fit_transformer_model, blend_models
from scenario import generate_scenarios, run_scenario_analysis
from exotics import american_call_baw, asian_option_geometric
from diagnostics import compute_errors, flag_outliers
from cloud_batch import run_parallel_local
from backtest import run_backtest

def dummy_func(x):
    return x+1

class TestIntegration(unittest.TestCase):
    def test_black_scholes_and_ml(self):
        S, T, r, sigma = 100, 0.25, 0.05, 0.2
        Ks = np.linspace(90, 110, 5)
        X = np.array([[S, K, T, r, sigma] for K in Ks])
        y = np.array([OptionPricing(S, K, T, r, sigma, 'call').black_scholes()[0] for K in Ks])
        model, scaler = fit_ml_model(X, y, epochs=2)
        preds = model(torch.tensor(scaler.transform(X), dtype=torch.float32)).detach().numpy().flatten()
        self.assertEqual(preds.shape, y.shape)
    def test_lstm_transformer(self):
        X_seq = np.random.rand(10, 5, 6)
        y = np.random.rand(10)
        fit_lstm_model(X_seq, y, epochs=2)
        fit_transformer_model(X_seq, y, epochs=2)
    def test_blend_models(self):
        S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
        op = OptionPricing(S, K, T, r, sigma, 'call')
        price, *_ = op.black_scholes()
        # Use at least 5 samples
        X = np.array([[S, K, T, r, sigma + i*0.01, 'call'] for i in range(5)])
        y = np.array([op.black_scholes()[0] for _ in range(5)])
        model, scaler = fit_ml_model(X[:, :5], y, epochs=2)
        # Blend only MLP for now
        blend = blend_models(X[:, :5], [(model, 'MLP')], scalers=[scaler])
        self.assertEqual(len(blend), len(y))
    def test_scenario_exotics(self):
        S, K, T, r, sigma = 100, 105, 0.25, 0.05, 0.2
        scenarios = generate_scenarios(S, sigma, r, T, n=2)
        df = pd.DataFrame({'strike': [K], 'type': ['call']})
        run_scenario_analysis(df, scenarios)
        american_call_baw(S, K, T, r, sigma)
        asian_option_geometric(S, K, T, r, sigma, 'call')
    def test_diagnostics_cloud_backtest(self):
        df = pd.DataFrame({'price': [10, 12], 'market_price': [10.5, 11.5], 'strike': [100, 105], 'T': [0.2, 0.3]})
        df_err, _, _ = compute_errors(df)
        flag_outliers(df_err)
        run_parallel_local(dummy_func, [(1,), (2,)])
        def pricing_func(row): return 10
        def strategy_func(row, price): return 'buy' if price < 11 else 'sell'
        hist = pd.DataFrame([{'dummy': 1} for _ in range(5)])
        run_backtest(pricing_func, hist, strategy_func)
    def test_dashboard_import(self):
        import dashboard

if __name__ == "__main__":
    unittest.main()
