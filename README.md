# Option Pricing Model Project

**GitHub Repo:** [https://github.com/MeridianAlgo/Option-Pricing-Research](https://github.com/MeridianAlgo/Option-Pricing-Research)

**Made with love by the MeridianAlgo Algorithmic Research Team (Quantum Meridian)**

A fully automated, quant-grade option pricing and research platform. Input a stock ticker, and the system fetches all data, computes prices using advanced models (Black-Scholes, Binomial, Monte Carlo, Heston, Merton, SABR, Variance Gamma, exotics), and provides ML-based pricing with PyTorch (MLP, LSTM, Transformer, blending, auto-retraining). Batch pricing, scenario analysis, full Greeks, risk metrics, RL for hedging, generative models, explainable AI dashboards, auto-documentation, research collaboration, and robust integration testing are all included.

## 🚀 Features

- **Automated Data Fetching**: Just provide a ticker, and all market data is fetched automatically (price, volatility, option chain, risk-free rate).
- **Advanced Pricing Models**: Black-Scholes, Binomial, Monte Carlo, Heston (stochastic volatility), Merton (jump diffusion), SABR, Variance Gamma, and more.
- **Exotic Options**: American, Asian, Barrier, Lookback, and more.
- **Machine Learning Pricing**: PyTorch-based neural networks (MLP, LSTM, Transformer), model blending, auto-retraining, and explainability (SHAP).
- **Reinforcement Learning**: RL agents for hedging and trading (scaffolded for future research).
- **Generative Models**: For volatility surface and scenario generation (scaffolded for future research).
- **Batch & Scenario Analysis**: Price entire option chains, run stress tests, and scenario analysis.
- **Full Greeks & Risk Metrics**: Delta, Gamma, Vega, Theta, Rho, Vomma, Vanna, VaR, CVaR, and more.
- **Diagnostics & Outlier Detection**: Error analysis, arbitrage checks, model fit stats, and robust integration tests.
- **Quant-Grade Reporting**: Export results to CSV, Excel, and generate interactive plots and diagnostics.
- **Explainable AI Dashboards**: Streamlit dashboard with session saving, sharing, annotation, and multi-user support.
- **Auto-Documentation & Collaboration**: Jupyter notebook template export, research automation, and collaboration tools.
- **Cloud/Distributed Computing**: Batch/scenario/Monte Carlo jobs on cloud or clusters (scaffolded for future research).
- **Extensible API**: Use as CLI, Python API, or batch mode. Modular for research and production.

## 📋 Requirements

- Python 3.7+
- NumPy
- SciPy
- Matplotlib
- Pandas
- yfinance
- requests
- torch
- scikit-learn
- streamlit
- pyyaml
- (Optional: shap, dask, ray, jupyter)

## 🛠️ Installation

1. Clone or download the project files
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Automated Pricing

```bash
python auto_pricer.py AAPL
```
- Fetches all data and prices all options for AAPL, saving results to CSV.

### Programmatic Usage

```python
from option_pricing import OptionPricing
from data import get_stock_price, get_option_chain, get_historical_prices, get_risk_free_rate

S = get_stock_price('AAPL')
K = 150
T = 0.25
r = get_risk_free_rate()
sigma = 0.2
option = OptionPricing(S, K, T, r, sigma, 'call')
price, delta, gamma, theta, vega, rho, vomma, vanna = option.black_scholes()
```

### Machine Learning Pricing

```python
from ml_model import fit_ml_model, fit_lstm_model, fit_transformer_model, blend_models
# X, y = ... # Prepare your features and target (option prices)
model, scaler = fit_ml_model(X, y)
predictions = model(torch.tensor(scaler.transform(X), dtype=torch.float32)).detach().numpy().flatten()
```

### Dashboard

```bash
streamlit run dashboard.py
```

### Batch/Cloud/Distributed
- See `config.yaml` and `cloud_batch.py` for batch and distributed job support.

## 📊 Models Implemented
- Black-Scholes (Analytical)
- Binomial Tree (Numerical)
- Monte Carlo (Statistical)
- Heston (Stochastic Volatility, placeholder)
- Merton (Jump Diffusion, placeholder)
- SABR, Variance Gamma (placeholders)
- American, Asian, Barrier, Lookback (exotics, partial/placeholder)
- PyTorch ML Models (MLP, LSTM, Transformer, blending)
- RL for hedging/trading (scaffolded)
- Generative models (scaffolded)

## 📈 Features for Quants
- Full Greeks (Delta, Gamma, Vega, Theta, Rho, Vomma, Vanna)
- Batch pricing for entire option chains
- Scenario and stress testing
- Model diagnostics and error analysis
- Export to CSV/Excel
- Interactive plots and surfaces
- Explainable AI dashboards
- Auto-documentation and research collaboration
- Robust integration and unit testing

## 📚 Mathematical Background
- See docstrings and code for model details and references.

---
## A Quick Disclaimer

This material is for informational purposes only and does not constitute financial, investment, legal, tax, or accounting advice. It is not intended to provide personalized recommendations or solicitations to buy or sell any securities or financial products.
Investing involves substantial risks, including the potential loss of principal. Market conditions, economic factors, and other variables can lead to volatility and losses. Past performance is not indicative of future results; historical returns do not guarantee similar outcomes.
Always consult a qualified financial advisor, attorney, or tax professional to assess your specific situation, risk tolerance, and objectives before making any investment decisions. We assume no liability for actions taken based on this information.

---
## Licenses 

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

---

**Made with ❤️ by Quantum Meridian (A MeridianAlgo Team)**

*Empowering the next generation of quantitative finance professionals through hands-on learning and practical implementation.*



