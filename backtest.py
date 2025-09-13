import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_backtest(pricing_func, hist_data, strategy_func, initial_cash=100000):
    """
    Run backtest of a strategy over historical data.
    pricing_func: function to price options
    hist_data: DataFrame with historical prices
    strategy_func: function to generate trades/signals
    Returns: DataFrame with portfolio value over time.
    """
    cash = initial_cash
    portfolio = []
    trades = []
    for date, row in hist_data.iterrows():
        price = pricing_func(row)
        signal = strategy_func(row, price)
        if signal == 'buy' and cash >= price:
            cash -= price
            portfolio.append({'date': date, 'price': price})
            trades.append({'date': date, 'action': 'buy', 'price': price})
        elif signal == 'sell' and portfolio:
            bought = portfolio.pop(0)
            cash += price
            trades.append({'date': date, 'action': 'sell', 'price': price})
    values = [initial_cash]
    for t in trades:
        if t['action'] == 'buy':
            values.append(values[-1] - t['price'])
        else:
            values.append(values[-1] + t['price'])
    return pd.DataFrame({'portfolio_value': values})

def generate_report(results_df):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['portfolio_value'])
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Portfolio Value')
    plt.grid(True, alpha=0.3)
    plt.show()

def export_notebook(template_path='backtest_template.ipynb'):
    # Placeholder: generate Jupyter notebook template
    notebook_json = '{\n "cells": [\n  {\n   "cell_type": "markdown",\n   "metadata": {},\n   "source": [\n    "# Backtest Research Template\\n",\n    "Add your research code here."\n   ]\n  }\n ],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 2\n}'
    with open(template_path, 'w') as f:
        f.write(notebook_json)
