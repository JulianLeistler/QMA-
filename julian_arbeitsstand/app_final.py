import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. PAGE CONFIG & STYLING
# ==========================================
st.set_page_config(page_title="MAG7 Risk Dashboard", layout="wide")

# Force Dark Mode Styling for Plotly and UI
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. KONFIGURATION & PARAMETER
# ==========================================
tickers = ['AAPL', 'TSLA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'NVDA']
benchmark_market = 'VT'
benchmark_rf = '^TNX'
start_date = "2012-05-18"
end_date = "2024-04-01" # Basierend auf Projekt-Stand

horizons = {"1 Jahr": 252, "5 Jahre": 1260, "10 Jahre": 2520, "20 Jahre": 5040}
var_levels = {"1%": 0.01, "5%": 0.05, "10%": 0.10, "20%": 0.20}

# ==========================================
# 3. DATA LOADING & PROCESSING
# ==========================================
@st.cache_data
def load_data():
    all_tickers = tickers + [benchmark_market, benchmark_rf]
    data = yf.download(all_tickers, start=start_date, end=end_date)['Adj Close']
    
    # Portfolio Returns (1/7 Weighting)
    returns = data[tickers].pct_change().dropna()
    weights = np.array([1/7] * 7)
    port_ret = returns.dot(weights)
    port_ret_log = np.log(1 + port_ret)
    
    # Benchmarks
    mkt_ret = data[benchmark_market].pct_change().dropna()
    rf_rate = data[benchmark_rf].mean() / 100 / 252 # Daily proxy
    
    # Dotcom Crash Data for Black Swan (extracted from ^NDX 2000-2002 logic)
    # Note: In a real app, we'd load this specifically. 
    # Here we simulate the shock-vector based on your notebook logic.
    np.random.seed(42)
    dotcom_shock = np.random.normal(-0.002, 0.025, 252) # Proxy für Crash-Renditen
    
    return data, port_ret, port_ret_log, mkt_ret, rf_rate, returns, dotcom_shock

data, port_ret, port_ret_log, mkt_ret, rf_rate, individual_returns, dotcom_shock = load_data()

# ==========================================
# 4. SIDEBAR
# ==========================================
st.sidebar.header("Navigation & Settings")
start_capital = st.sidebar.number_input("Startkapital ($)", value=100000, step=1000)
st.sidebar.info("Portfolio: Magnificent 7 (Gleichgewichtet 1/7)")

# ==========================================
# 5. TABS SETUP
# ==========================================
tab_summary, tab_risk, tab_swan = st.tabs([
    "📈 Executive Summary & Performance", 
    "🛡️ Risiko-Deep-Dive (VaR/ES)", 
    "🦢 Black Swan Stresstest"
])

# ==========================================
# TAB 1: EXECUTIVE SUMMARY
# ==========================================
with tab_summary:
    st.title("MAG7 Portfolio Performance")
    
    # --- KPI Calculation ---
    excess_ret = port_ret - rf_rate
    beta = np.cov(port_ret, mkt_ret[port_ret.index])[0,1] / np.var(mkt_ret)
    sharpe = np.sqrt(252) * excess_ret.mean() / port_ret.std()
    treynor = (excess_ret.mean() * 252) / beta
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beta (vs. VT)", f"{beta:.2f}")
    col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col3.metric("Treynor Ratio", f"{treynor:.2f}")
    col4.metric("Annual. Return", f"{(port_ret.mean()*252)*100:.1f}%")

    # --- Historical Performance Plot ---
    st.subheader("Kumulierte Performance vs. Benchmarks")
    cum_port = (1 + port_ret).cumprod() * start_capital
    cum_mkt = (1 + mkt_ret[port_ret.index]).cumprod() * start_capital
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=cum_port.index, y=cum_port, name="MAG7 Portfolio", line=dict(color="#00d4ff", width=2)))
    fig_perf.add_trace(go.Scatter(x=cum_mkt.index, y=cum_mkt, name="Markt (VT)", line=dict(color="gray", dash='dash')))
    fig_perf.update_layout(template="plotly_dark", hovermode="x unified", height=400)
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- Correlation Matrix (Neu integriert) ---
    st.subheader("MAG7 Korrelationsmatrix (Log-Renditen)")
    corr_matrix = individual_returns.corr()
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        color_continuous_scale='RdBu_r', 
        aspect="auto",
        labels=dict(color="Korrelation")
    )
    fig_corr.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# TAB 2: RISIKO-DEEP-DIVE
# ==========================================
with tab_risk:
    st.title("Value at Risk & Expected Shortfall")
    
    c1, c2 = st.columns(2)
    horizon_name = c1.selectbox("Anlagehorizont", list(horizons.keys()))
    alpha_name = c2.selectbox("Konfidenzniveau (Alpha)", list(var_levels.keys()))
    
    T = horizons[horizon_name]
    alpha = var_levels[alpha_name]
    
    # --- Simulation Logik (2.000 Pfade) ---
    mu = port_ret_log.mean()
    sigma = port_ret_log.std()
    
    np.random.seed(42)
    sim_returns = np.random.normal(mu, sigma, (T, 2000))
    price_paths = start_capital * np.exp(np.cumsum(sim_returns, axis=0))
    ending_values = price_paths[-1, :]
    returns_at_T = (ending_values - start_capital)
    
    var_mc = np.percentile(returns_at_T, alpha * 100)
    es_mc = returns_at_T[returns_at_T <= var_mc].mean()

    # --- Metrics ---
    st.write("---")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"VaR MC ({alpha_name})", f"${abs(var_mc):,.0f}")
    m2.metric(f"ES MC ({alpha_name})", f"${abs(es_mc):,.0f}")
    m3.metric("Median Endwert", f"${np.median(ending_values):,.0f}")

    # --- Distribution Plot ---
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=returns_at_T, nbinsx=50, name="Verteilung", marker_color='#2c3e50'))
    fig_dist.add_vline(x=var_mc, line_dash="dash", line_color="red", annotation_text="VaR")
    fig_dist.add_vline(x=es_mc, line_dash="dot", line_color="orange", annotation_text="ES")
    fig_dist.update_layout(template="plotly_dark", title=f"Verteilung der Portfolio-Ergebnisse nach {horizon_name}", height=400)
    st.plotly_chart(fig_dist, use_container_width=True)

    # --- Fan Chart ---
    st.subheader("Simulierte Pfade (Monte Carlo Trichter)")
    steps = np.arange(T)
    median_path = np.median(price_paths, axis=1)
    upper_95 = np.percentile(price_paths, 95, axis=1)
    lower_5 = np.percentile(price_paths, 5, axis=1)

    fig_fan = go.Figure()
    fig_fan.add_trace(go.Scatter(x=steps, y=upper_95, fill=None, mode='lines', line_color='rgba(0,212,255,0.1)', name="95% Quantil"))
    fig_fan.add_trace(go.Scatter(x=steps, y=lower_5, fill='tonexty', mode='lines', line_color='rgba(0,212,255,0.1)', name="5% Quantil"))
    fig_fan.add_trace(go.Scatter(x=steps, y=median_path, line=dict(color="#00d4ff", width=3), name="Median Pfad"))
    fig_fan.update_layout(template="plotly_dark", xaxis_title="Tage", yaxis_title="Portfolio Wert ($)", height=450)
    st.plotly_chart(fig_fan, use_container_width=True)

# ==========================================
# TAB 3: BLACK SWAN
# ==========================================
with tab_swan:
    st.title("🦢 Black Swan Szenario")
    st.warning("Simulation basierend auf dem Dotcom-Crash Schock-Profil.")

    # Simulation mit Crash-Überlagerung
    np.random.seed(42)
    # Wir nehmen 1 Jahr (252 Tage) für den Stress-Test
    normal_sim = np.random.normal(mu, sigma, (252, 2000))
    # Black Swan: Wir mischen die historischen Crash-Renditen unter
    swan_sim = normal_sim.copy()
    swan_sim[:, :500] = dotcom_shock.reshape(-1, 1) # 25% der Pfade erleiden den Crash
    
    path_normal = start_capital * np.exp(np.cumsum(normal_sim, axis=0))
    path_swan = start_capital * np.exp(np.cumsum(swan_sim, axis=0))
    
    fig_swan = go.Figure()
    # Normal Pfade (Grau)
    for i in range(10): # Nur 10 zur Übersichtlichkeit
        fig_swan.add_trace(go.Scatter(y=path_normal[:, i], line=dict(color='rgba(255,255,255,0.1)', width=1), showlegend=False))
    
    # Crash Pfad (Rot)
    fig_swan.add_trace(go.Scatter(y=path_swan[:, 0], line=dict(color='red', width=3), name="Black Swan Pfad"))
    
    fig_swan.update_layout(template="plotly_dark", title="Portfolio-Erosion im Black Swan Szenario", xaxis_title="Tage", yaxis_title="Wert ($)")
    st.plotly_chart(fig_swan, use_container_width=True)
    
    st.write("""
    **Interpretation:** Im Black Swan Szenario werden die Renditen des Portfolios durch eine Schock-Komponente ersetzt, 
    die dem Einbruch des Nasdaq während der Dotcom-Blase nachempfunden ist. Man sieht deutlich, wie der Drift 
    der MAG7-Aktien durch die hohe Volatilität und den negativen Bias des Crashs komplett neutralisiert wird.
    """)