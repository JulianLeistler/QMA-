import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# 1. PAGE CONFIG & DARK MODE STYLING
# ==========================================
st.set_page_config(page_title="MAG7 Risiko Dashboard", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    div[data-testid="stMetricValue"] { color: #00d4ff; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. KONFIGURATION (WIE IM PROJEKT)
# ==========================================
tickers = ['AAPL', 'TSLA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'NVDA']
benchmark_world_ticker = 'VT'
benchmark_rf_ticker = '^TNX'
start_date = "2012-05-18"
end_date = "2024-04-01"

horizons = {"1 Jahr": 252, "5 Jahre": 1260, "10 Jahre": 2520, "20 Jahre": 5040}
var_levels_ui = {"1%": 0.01, "5%": 0.05, "10%": 0.10, "20%": 0.20}

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    all_assets = tickers + [benchmark_world_ticker, benchmark_rf_ticker]
    # WICHTIG: 'Close' nutzen um KeyError 'Adj Close' zu vermeiden
    raw_data = yf.download(all_assets, start=start_date, end=end_date, auto_adjust=False)['Close']
    
    # Renditen berechnen
    returns_df = raw_data[tickers].pct_change().dropna()
    weights = np.array([1/7] * 7)
    port_ret = returns_df.dot(weights)
    port_ret_log = np.log(1 + port_ret)
    
    mkt_ret = raw_data[benchmark_world_ticker].pct_change().dropna()
    rf_daily = (raw_data[benchmark_rf_ticker].mean() / 100) / 252
    
    # Black Swan Daten: Dotcom Crash (^NDX 2000-2002)
    swan_data = yf.download("^NDX", start="2000-01-01", end="2002-12-31", auto_adjust=False)['Close']
    swan_ret_log = np.log(swan_data / swan_data.shift(1)).dropna()
    
    return raw_data, port_ret, port_ret_log, mkt_ret, rf_daily, returns_df, swan_ret_log

data, port_ret, port_ret_log, mkt_ret, rf_daily, ind_ret, swan_ret_log = load_data()

# ==========================================
# 4. RISIKO FUNKTIONEN (AUS DEM NOTEBOOK)
# ==========================================
def calculate_historical_risk(log_ret, capital, alpha, T):
    hist_samples = []
    np.random.seed(42)
    for _ in range(2000):
        start = np.random.randint(0, len(log_ret) - T)
        path = log_ret.iloc[start:start+T].sum()
        hist_samples.append(capital * (np.exp(path) - 1))
    var = np.percentile(hist_samples, alpha * 100)
    es = np.mean([s for s in hist_samples if s <= var])
    return var, es

def calculate_gaussian_risk(log_ret, capital, alpha, T):
    mu = log_ret.mean()
    sigma = log_ret.std()
    z = stats.norm.ppf(alpha)
    var = capital * (np.exp(mu * T + z * sigma * np.sqrt(T)) - 1)
    es = capital * (np.exp(mu * T) * (stats.norm.cdf(z - sigma * np.sqrt(T)) / alpha) - 1)
    return var, es

def calculate_lognormal_risk(log_ret, capital, alpha, T):
    # Bei Log-Renditen ist die Lognormal-Transformation praktisch identisch zur Gauss-Transformation 
    # auf den Endwert, wird hier separat gehalten für die saubere Methodentrennung.
    return calculate_gaussian_risk(log_ret, capital, alpha, T)

def calculate_monte_carlo_risk(log_ret, capital, alpha, T, simulations=2000, black_swan=False, crash_data=None):
    mu = log_ret.mean()
    sigma = log_ret.std()
    np.random.seed(42)
    
    sim_returns = np.random.normal(mu, sigma, (T, simulations))
    
    if black_swan and crash_data is not None:
        # Ersetze ca. 25% der Pfade mit dem historischen Crash-Verlauf
        num_crashes = int(simulations * 0.25)
        # Crash Data samplen
        crash_sample = crash_data.sample(T, replace=True).values
        for i in range(num_crashes):
            sim_returns[:, i] = crash_sample
            
    paths = capital * np.exp(np.cumsum(sim_returns, axis=0))
    final_returns = paths[-1, :] - capital
    
    var = np.percentile(final_returns, alpha * 100)
    es = np.mean(final_returns[final_returns <= var])
    return var, es, final_returns, paths

# ==========================================
# 5. UI LAYOUT & TABS
# ==========================================
st.sidebar.title("Dashboard Steuerung")
start_capital = st.sidebar.number_input("Startkapital ($)", value=100000, step=5000)

tab1, tab2, tab3 = st.tabs(["📊 Übersicht & Performance", "🛡️ Risiko-Analyse (VaR/ES)", "🦢 Black Swan Stresstest"])

# --- TAB 1: ÜBERSICHT ---
with tab1:
    st.header("Executive Summary")
    
    # KPIs
    common_idx = port_ret.index.intersection(mkt_ret.index)
    beta = np.cov(port_ret.loc[common_idx], mkt_ret.loc[common_idx])[0, 1] / np.var(mkt_ret.loc[common_idx])
    excess_ret = port_ret.mean() - rf_daily
    sharpe = (excess_ret * 252) / (port_ret.std() * np.sqrt(252))
    treynor = (excess_ret * 252) / beta
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beta (vs. VT)", f"{beta:.2f}")
    c2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c3.metric("Treynor Ratio", f"{treynor:.2f}")
    c4.metric("Anual. Rendite", f"{(port_ret.mean()*252*100):.1f}%")
    
    # Performance Plot
    st.subheader("Kumulierte Performance")
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=port_ret.index, y=(1+port_ret).cumprod()*start_capital, name="MAG7 Portfolio", line=dict(color="#00d4ff")))
    fig_perf.add_trace(go.Scatter(x=mkt_ret.index, y=(1+mkt_ret).cumprod()*start_capital, name="Markt (VT)", line=dict(color="gray", dash="dash")))
    fig_perf.update_layout(template="plotly_dark", height=400, hovermode="x unified")
    st.plotly_chart(fig_perf, use_container_width=True)
    
    # Korrelation
    st.subheader("MAG7 Korrelationsmatrix")
    fig_corr = px.imshow(ind_ret.corr(), text_auto=".2f", color_continuous_scale="RdBu_r")
    fig_corr.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# --- TAB 2: RISIKO-ANALYSE ---
with tab2:
    st.header("Modellgestützte Risiko-Einschätzung")
    
    col_a, col_b = st.columns(2)
    sel_h = col_a.selectbox("Anlagehorizont", list(horizons.keys()))
    sel_v = col_b.selectbox("VaR-Konfidenz", list(var_levels_ui.keys()))
    
    T = horizons[sel_h]
    alpha = var_levels_ui[sel_v]
    
    # Berechnungen aufrufen
    var_hist, es_hist = calculate_historical_risk(port_ret_log, start_capital, alpha, T)
    var_gauss, es_gauss = calculate_gaussian_risk(port_ret_log, start_capital, alpha, T)
    var_log, es_log = calculate_lognormal_risk(port_ret_log, start_capital, alpha, T)
    var_mc, es_mc, mc_rets, mc_paths = calculate_monte_carlo_risk(port_ret_log, start_capital, alpha, T)
    
    # Tabelle anzeigen
    res_data = {
        "Methode": ["Historisch", "Gaußsch", "Lognormal", "Monte Carlo"],
        "VaR ($)": [var_hist, var_gauss, var_log, var_mc],
        "ES ($)": [es_hist, es_gauss, es_log, es_mc]
    }
    df_res = pd.DataFrame(res_data)
    st.table(df_res.style.format({"VaR ($)": "{:,.0f}", "ES ($)": "{:,.0f}"}))
    
    # Verteilung
    st.subheader(f"Verteilung der Endwerte (Monte Carlo - {sel_h})")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=mc_rets + start_capital, nbinsx=50, name="Verteilung", marker_color='#2c3e50'))
    fig_hist.add_vline(x=start_capital + var_mc, line_dash="dash", line_color="red", annotation_text="VaR")
    fig_hist.add_vline(x=start_capital + es_mc, line_dash="dot", line_color="orange", annotation_text="ES")
    fig_hist.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# --- TAB 3: BLACK SWAN ---
with tab3:
    st.header("Stresstest: Dotcom-Szenario")
    st.info("Simulation eines schweren Marktschocks basierend auf den Nasdaq-Renditen der Jahre 2000-2002.")
    
    days_swan = horizons["1 Jahr"] # Fester Horizont für Stresstest
    swan_lvl_name = st.selectbox("Auswahl VaR-Level für Stresstest", list(var_levels_ui.keys()), key="var_swan")
    alpha_swan = var_levels_ui[swan_lvl_name]
    
    mc_var_norm, mc_es_norm, _, paths_norm = calculate_monte_carlo_risk(
        port_ret_log, start_capital, alpha_swan, days_swan, simulations=2000, black_swan=False
    )
    mc_var_swan, mc_es_swan, _, paths_swan = calculate_monte_carlo_risk(
        port_ret_log, start_capital, alpha_swan, days_swan, simulations=2000, black_swan=True, crash_data=swan_ret_log
    )
    
    st.write("---")
    c1, c2 = st.columns(2)
    c1.subheader("Ohne Black Swan ('Normal')")
    c1.metric("VaR ($)", f"${mc_var_norm:,.0f}")
    c1.metric("ES ($)", f"${mc_es_norm:,.0f}")
    
    c2.subheader("Mit Black Swan")
    c2.metric("VaR ($)", f"${mc_var_swan:,.0f}")
    c2.metric("ES ($)", f"${mc_es_swan:,.0f}")

    st.write("---")
    st.subheader("Pfad-Visualisierung: Normal vs. Black Swan")
    
    fig_swan = go.Figure()
    # Zeichne ein paar normale Pfade (Grau)
    for i in range(10):
        fig_swan.add_trace(go.Scatter(y=paths_norm[:, i], line=dict(color='rgba(255,255,255,0.1)'), showlegend=False))
    
    # Zeichne den schlimmsten Swan-Pfad (Rot)
    worst_swan_idx = np.argmin(paths_swan[-1, :])
    fig_swan.add_trace(go.Scatter(y=paths_swan[:, worst_swan_idx], name="Worst-Case Black Swan Pfad", line=dict(color='red', width=3)))
    
    fig_swan.update_layout(template="plotly_dark", yaxis_title="Portfolio Wert ($)", xaxis_title="Tage (1 Jahr Horizont)", height=450)
    st.plotly_chart(fig_swan, use_container_width=True)