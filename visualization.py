import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
import plotly.offline as py


def plot_surface(df, x, y, z, title="Option Price Surface", zlabel="Price"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(sorted(df[x].unique()), sorted(df[y].unique()))
    Z = df.pivot_table(index=y, columns=x, values=z).values
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

def plot_greeks_surface(df, greek, x='strike', y='T', title=None):
    if not title:
        title = f"{greek} Surface"
    plot_surface(df, x, y, greek, title=title, zlabel=greek)

def plot_implied_vol_surface(df, x='strike', y='T', z='impliedVolatility', title="Implied Volatility Surface"):
    plot_surface(df, x, y, z, title=title, zlabel="Implied Volatility")

def plot_residuals(y_true, y_pred, title="Residuals Histogram"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue')
    plt.title(title)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_model_comparison(results_dict):
    plt.figure(figsize=(10, 6))
    for label, values in results_dict.items():
        plt.plot(values['x'], values['y'], label=label)
    plt.xlabel("Strike")
    plt.ylabel("Option Price")
    plt.title("Model Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_all_greeks_surfaces(df, x='strike', y='T'):
    """
    Plot 3D surfaces for all major Greeks in the DataFrame.
    """
    greeks = ['delta', 'gamma', 'vega', 'theta', 'rho', 'vomma', 'vanna']
    for greek in greeks:
        if greek in df.columns:
            plot_greeks_surface(df, greek, x=x, y=y, title=f"{greek.title()} Surface")

def plot_pnl_distribution(pnl, title="P&L Distribution"):
    """
    Plot histogram of P&L distribution.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(pnl, bins=40, alpha=0.7, color='green')
    plt.title(title)
    plt.xlabel("P&L")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.show()

def export_html_report(df, filename="report.html"):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns)),
        cells=dict(values=[df[col] for col in df.columns])
    )])
    py.plot(fig, filename=filename, auto_open=False)

def export_excel(df, filename="report.xlsx"):
    df.to_excel(filename, index=False)
