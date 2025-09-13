import yfinance as yf
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

FRED_API = "https://api.stlouisfed.org/fred/series/observations"
FRED_SERIES = "DGS3MO"  # 3-Month Treasury Bill
FRED_KEY = ""  # Optional: Add your FRED API key here


def get_stock_price(ticker):
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")['Close'].iloc[-1]
    return float(price)


def get_option_chain(ticker):
    stock = yf.Ticker(ticker)
    expiries = stock.options
    chain = {}
    for expiry in expiries:
        opt = stock.option_chain(expiry)
        chain[expiry] = {
            'calls': opt.calls,
            'puts': opt.puts
        }
    return chain


def get_historical_prices(ticker, period="1y"):  # 1 year by default
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist['Close']


def get_risk_free_rate():
    # Get most recent 3-month T-bill rate from FRED
    params = {
        'series_id': FRED_SERIES,
        'api_key': FRED_KEY,
        'file_type': 'json',
        'sort_order': 'desc',
        'limit': 1
    }
    try:
        resp = requests.get(FRED_API, params=params)
        data = resp.json()
        rate = float(data['observations'][0]['value']) / 100
        return rate
    except Exception:
        # Fallback: use 0.05 (5%)
        return 0.05


def get_implied_volatility(option_row):
    # yfinance provides 'impliedVolatility' for some options
    if 'impliedVolatility' in option_row:
        return float(option_row['impliedVolatility'])
    # Fallback: estimate from historical volatility
    return None


def estimate_historical_volatility(prices, window=252):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return np.std(log_returns) * np.sqrt(window)
