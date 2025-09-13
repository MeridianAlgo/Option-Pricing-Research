"""
Streamlit Dashboard for Option Pricing Analytics
Run with: streamlit run dashboard.py
Features: session saving, sharing, annotation, multi-user (scaffold), notebook download.
"""
import streamlit as st
import pandas as pd
import numpy as np
from visualization import plot_surface, plot_greeks_surface, plot_all_greeks_surfaces
from scenario import generate_scenarios, run_scenario_analysis, plot_scenario_surface
from diagnostics import plot_error_surface, plot_residuals_vs_features
import uuid
import os

st.title("Option Pricing Analytics Dashboard")

# Session management
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = str(uuid.uuid4())

uploaded = st.file_uploader("Upload Option Prices CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Data Preview:", df.head())
    st.subheader("Surface Plots")
    surface_type = st.selectbox("Surface", ["price", "delta", "gamma", "vega", "theta", "rho", "vomma", "vanna"])
    if surface_type in df.columns:
        st.write(f"{surface_type.title()} Surface")
        plot_surface(df, x='strike', y='T', z=surface_type, title=f"{surface_type.title()} Surface")
    st.subheader("Scenario Analysis")
    base_S = float(st.number_input("Spot", value=float(df['S'].iloc[0])))
    base_sigma = float(st.number_input("Volatility", value=float(df['sigma'].mean())))
    base_r = float(st.number_input("Rate", value=float(df['r'].mean())))
    base_T = float(st.number_input("Time to Expiry", value=float(df['T'].mean())))
    if st.button("Run Scenario Analysis"):
        scenarios = generate_scenarios(base_S, base_sigma, base_r, base_T, n=5, pct_range=0.2)
        scenario_df = run_scenario_analysis(df.head(5), scenarios)
        st.write(scenario_df.head())
        plot_scenario_surface(scenario_df, scenario_param='scenario_S', greek='price')
    st.subheader("Diagnostics")
    if 'residual' in df.columns:
        plot_error_surface(df, x='strike', y='T', z='residual')
        plot_residuals_vs_features(df, features=['strike', 'T'])
    # Annotation
    annotation = st.text_area("Add annotation for this session:")
    if st.button("Save Annotation"):
        with open(f"annotation_{st.session_state['session_id']}.txt", 'w') as f:
            f.write(annotation)
        st.success("Annotation saved.")
    # Session saving
    if st.button("Save Session"):
        df.to_csv(f"session_{st.session_state['session_id']}.csv", index=False)
        st.success("Session saved.")
    # Session sharing (scaffold)
    st.write(f"Session ID: {st.session_state['session_id']}")
    # Multi-user (scaffold): could use database or file-based sharing
    # Notebook download
    if st.button("Download Jupyter Notebook Template"):
        from backtest import export_notebook
        export_notebook(f"backtest_{st.session_state['session_id']}.ipynb")
        with open(f"backtest_{st.session_state['session_id']}.ipynb", "rb") as f:
            st.download_button("Download Notebook", f.read(), file_name="backtest_template.ipynb")
    st.download_button("Download CSV", df.to_csv(index=False), file_name="option_report.csv")
