import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_errors(df, market_col='market_price', model_col='price'):
    """
    Compute residuals, RMSE, MAE between model and market prices.
    Adds 'residual' column to df.
    """
    df = df.copy()
    df['residual'] = df[model_col] - df[market_col]
    rmse = np.sqrt(np.mean(df['residual']**2))
    mae = np.mean(np.abs(df['residual']))
    return df, rmse, mae

def flag_outliers(df, threshold=2.5):
    """
    Flag options with residuals > threshold*std or arbitrage (negative price, etc.)
    Adds 'outlier' column to df.
    """
    std = df['residual'].std()
    df['outlier'] = np.abs(df['residual']) > threshold * std
    df['arbitrage'] = df['price'] < 0
    return df

def plot_error_surface(df, x='strike', y='T', z='residual', title="Error Surface"):
    from visualization import plot_surface
    plot_surface(df, x, y, z, title=title, zlabel="Residual")

def plot_residuals_vs_features(df, features=['strike', 'T']):
    for feat in features:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[feat], df['residual'], alpha=0.6)
        plt.xlabel(feat)
        plt.ylabel('Residual')
        plt.title(f'Residuals vs {feat}')
        plt.grid(True, alpha=0.3)
        plt.show()

def print_model_fit_stats(rmse, mae, outlier_count, total):
    print(f"Model Fit: RMSE={rmse:.4f}, MAE={mae:.4f}, Outliers={outlier_count}/{total}")
