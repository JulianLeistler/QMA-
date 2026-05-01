import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. FUNKTIONEN (Logik & Visualisierung)
# ==========================================

def calculate_monte_carlo_risk_blackswan(portfolio_returns, start_capital=100000, var_level=0.05, days=252, simulations=10000, black_swan=False, crash_data=None):
    """Führt die Monte-Carlo-Simulation durch."""
    np.random.seed(42) # Für reproduzierbare Ergebnisse
    mu = np.mean(portfolio_returns)
    sigma = np.std(portfolio_returns)
    
    daily_returns = np.random.normal(loc=mu, scale=sigma, size=(days, simulations))
    
    if black_swan and crash_data is not None:
        crash_length = len(crash_data)
        if crash_length > 0 and days > crash_length:
            start_idx = np.random.randint(0, days - crash_length)
            for i in range(simulations):
                daily_returns[start_idx:start_idx+crash_length, i] = crash_data.values

    portfolio_paths = np.zeros_like(daily_returns)
    portfolio_paths[0] = start_capital * (1 + daily_returns[0])
    for t in range(1, days):
        portfolio_paths[t] = portfolio_paths[t-1] * (1 + daily_returns[t])
        
    final_values = portfolio_paths[-1]
    sorted_returns = np.sort(final_values - start_capital)
    
    var_index = int(simulations * var_level)
    var = sorted_returns[var_index]
    es = np.mean(sorted_returns[:var_index])
    
    return var, es, final_values, portfolio_paths

def plot_monte_carlo_fan_chart(portfolio_paths, start_capital, var_level=0.05, title="Monte Carlo Fan-Chart"):
    days = portfolio_paths.shape[0]
    lower_pct = var_level * 100
    upper_pct = (1 - var_level) * 100
    
    median_path = np.median(portfolio_paths, axis=1)
    lower_bound = np.percentile(portfolio_paths, lower_pct, axis=1)
    upper_bound = np.percentile(portfolio_paths, upper_pct, axis=1)

    x_axis = np.arange(1, days + 1)
    x_axis = np.insert(x_axis, 0, 0)
    median_path = np.insert(median_path, 0, start_capital)
    lower_bound = np.insert(lower_bound, 0, start_capital)
    upper_bound = np.insert(upper_bound, 0, start_capital)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_axis, y=lower_bound, mode='lines', line=dict(color='rgba(231, 76, 60, 0.8)', width=1), name=f'Worst Case ({lower_pct:g}% Quantil)'))
    fig.add_trace(go.Scatter(x=x_axis, y=median_path, mode='lines', line=dict(color='rgb(52, 152, 219)', width=3), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)', name='Median Pfad'))
    fig.add_trace(go.Scatter(x=x_axis, y=upper_bound, mode='lines', line=dict(color='rgba(46, 204, 113, 0.8)', width=1), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)', name=f'Best Case ({upper_pct:g}% Quantil)'))

    fig.add_hline(y=start_capital, line_dash="dash", line_color="rgba(255, 255, 255, 0.5)", annotation_text="Startkapital")
    fig.update_layout(title=title, xaxis_title="Handelstage", yaxis_title="Portfolio-Wert ($)", template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_black_swan_comparison(paths_normal, paths_swan, start_capital=100000, title="Median-Vergleich: Normal vs. Black Swan"):
    days = paths_normal.shape[0]
    median_normal = np.median(paths_normal, axis=1)
    median_swan = np.median(paths_swan, axis=1)
    
    x_axis = np.arange(1, days + 1)
    x_axis = np.insert(x_axis, 0, 0)
    median_normal = np.insert(median_normal, 0, start_capital)
    median_swan = np.insert(median_swan, 0, start_capital)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=median_normal, mode='lines', line=dict(color='rgb(52, 152, 219)', width=3), name='Median (Ohne Crash)'))
    fig.add_trace(go.Scatter(x=x_axis, y=median_swan, mode='lines', line=dict(color='rgb(231, 76, 60)', width=3, dash='dash'), name='Median (Mit DotCom-Crash)'))
    fig.add_hline(y=start_capital, line_dash="dot", line_color="rgba(255, 255, 255, 0.3)", annotation_text="Startkapital")
    
    fig.update_layout(title=title, xaxis_title="Handelstage", yaxis_title="Portfolio-Wert ($)", template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ==========================================
# 2. SEITEN-KONFIGURATION & CACHING
# ==========================================
st.set_page_config(page_title="MAG7 Risiko-Dashboard", layout="wide")

# Dummy-Daten generieren (Falls du hier deine eigenen Daten nutzt, diese Zeilen ersetzen!)
np.random.seed(99)
dotcom_returns = pd.Series(np.random.normal(-0.02, 0.05, 30)) 
portfolio_returns_log = pd.Series(np.random.normal(0.001, 0.015, 2520))

@st.cache_data
def get_cached_monte_carlo_paths(capital, days, simulations=10000):
    _, _, _, pf_paths = calculate_monte_carlo_risk_blackswan(
        portfolio_returns_log, capital, 0.05, days, simulations, black_swan=False
    )
    return pf_paths

@st.cache_data
def get_cached_black_swan_paths(capital, days, simulations=10000):
    _, _, _, pf_paths = calculate_monte_carlo_risk_blackswan(
        portfolio_returns_log, capital, 0.05, days, simulations, 
        black_swan=True, crash_data=dotcom_returns
    )
    return pf_paths

var_level_mapping = {"1 %": 0.01, "5 %": 0.05, "10 %": 0.10, "20 %": 0.20}

# ==========================================
# 3. SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Parameter")
    start_capital = st.number_input("Startkapital ($)", min_value=10000, max_value=10000000, value=100000, step=10000)

# ==========================================
# 4. HAUPTSEITE (Reiter)
# ==========================================
st.title("Magnificent 7: Risiko - Dashboard")

tab_uebersicht, tab_1j, tab_5j, tab_10j, tab_blackswan = st.tabs([
    "Übersicht", "1-Jahres-Risiko", "5-Jahres-Risiko", "10-Jahres-Risiko", "Black-Swan-Sim"
])

# --- REITER 0: ÜBERSICHT ---
with tab_uebersicht:
    st.header("Performance KPIs")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beta", "1.15", help="Sensitivität gegenüber dem Markt.")
    col2.metric("Sharpe Ratio", "1.82", help="Überrendite pro Einheit Gesamtrisiko.")
    col3.metric("Roy's Safety First", "1.05", help="Risiko, den Markt zu underperformen.")
    col4.metric("Treynor Ratio", "0.2450", help="Überrendite pro Einheit Marktrisiko.")
    st.markdown("---")
    st.info("📊 Platzhalter: Historischer Chart in voller Breite")
    st.markdown("---")
    st.subheader("Vollständige Risikomatrix")
    st.info("🧮 Platzhalter: Deine große 'core_results' Tabelle hier einfügen")

# --- REITER 1: 1-Jahres-Risiko ---
with tab_1j:
    paths_1j = get_cached_monte_carlo_paths(start_capital, 252)
    col_varA, col_varB = st.columns(2)
    
    with col_varA:
        st.subheader("Auswahl VaR-Level - A")
        var_a_str = st.selectbox("Level A", list(var_level_mapping.keys()), index=1, key="var_a_1j") 
        var_a_val = var_level_mapping[var_a_str]
        st.info(f"🧮 Tabelle: VaR und ES für {var_a_str}")
        fig_fan_a = plot_monte_carlo_fan_chart(paths_1j, start_capital, var_level=var_a_val, title=f"Monte Carlo ({var_a_str} VaR)")
        st.plotly_chart(fig_fan_a, use_container_width=True)
        
    with col_varB:
        st.subheader("Auswahl VaR-Level - B")
        var_b_str = st.selectbox("Level B", list(var_level_mapping.keys()), index=0, key="var_b_1j") 
        var_b_val = var_level_mapping[var_b_str]
        st.info(f"🧮 Tabelle: VaR und ES für {var_b_str}")
        fig_fan_b = plot_monte_carlo_fan_chart(paths_1j, start_capital, var_level=var_b_val, title=f"Monte Carlo ({var_b_str} VaR)")
        st.plotly_chart(fig_fan_b, use_container_width=True)

# --- REITER 5 & 10 JAHRE ---
with tab_5j: st.info("Platzhalter für 5 Jahre (1260 Tage) - Analog zu 1 Jahr aufbauen")
with tab_10j: st.info("Platzhalter für 10 Jahre (2520 Tage) - Analog zu 1 Jahr aufbauen")

# --- REITER: BLACK SWAN ---
with tab_blackswan:
    st.markdown("**Das Black-Swan-Event** simuliert den VaR & ES auf Basis eines zufällig eintretenden Crashes.")
    
    bs_horizon_years = st.selectbox("Auswahl Anlagehorizont (Jahre)", [1, 5, 10, 20], index=1)
    bs_horizon_days = bs_horizon_years * 252
    
    paths_normal = get_cached_monte_carlo_paths(start_capital, bs_horizon_days)
    paths_swan = get_cached_black_swan_paths(start_capital, bs_horizon_days)
    
    st.markdown("---")
    st.subheader("Direkter Vergleich der Median-Rendite")
    fig_comparison = plot_black_swan_comparison(paths_normal, paths_swan, start_capital)
    st.plotly_chart(fig_comparison, use_container_width=True)
    st.markdown("---")
    
    col_normal, col_swan = st.columns(2)
    with col_normal:
        st.subheader("Normal (Ohne Black Swan)")
        st.info("🧮 Tabelle VaR & ES")
        fig_normal = plot_monte_carlo_fan_chart(paths_normal, start_capital, var_level=0.05, title="Normal-Szenario")
        st.plotly_chart(fig_normal, use_container_width=True)
        
    with col_swan:
        st.subheader("Black Swan (DotCom Crash)")
        st.info("🧮 Tabelle VaR & ES")
        fig_swan = plot_monte_carlo_fan_chart(paths_swan, start_capital, var_level=0.05, title="Black Swan Szenario")
        st.plotly_chart(fig_swan, use_container_width=True)