import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ==========================================
# 1. PAGE CONFIG & SETUP
# ==========================================
st.set_page_config(page_title="MAG7 Risiko Dashboard", layout="wide")

# Konfiguration
tickers = ['AAPL', 'TSLA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'NVDA']
benchmark_world_ticker = 'VT'
benchmark_risk_free_ticker = '^TNX'
start_date = "2012-05-18"
end_date = "2026-04-01"

horizons = {
    "1 Jahr": 252,
    "5 Jahre": 1260,
    "10 Jahre": 2520,
    "20 Jahre": 5040,
}

var_levels_ui = {
    "1 %": 0.01,
    "5 %": 0.05,
    "10 %": 0.10,
    "20 %": 0.20,
}

monte_carlo_simulations = 10_000
bootstrap_simulations = 10_000
monte_carlo_seed = 2
bootstrap_seed = 1

# ==========================================
# 2. DATA LOADING (CACHED)
# ==========================================
@st.cache_data
def load_data():
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
    bench_world = yf.download(benchmark_world_ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
    bench_rf = yf.download(benchmark_risk_free_ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
    dotcom = yf.download('^NDX', start="2000-04-01", end="2002-10-31", auto_adjust=False, progress=False)['Adj Close']
    
    # Berechnungen
    dotcom_log_ret = np.log(dotcom / dotcom.shift(1)).dropna().values
    returns_discrete = data.pct_change().dropna()
    returns_log = np.log(data / data.shift(1)).dropna()
    weights = np.array([1/len(tickers)] * len(tickers))
    port_ret_discrete = returns_discrete.dot(weights)
    port_ret_log = returns_log.dot(weights)
    
    return port_ret_discrete, port_ret_log, bench_world, bench_rf, dotcom_log_ret

port_ret_discrete, port_ret_log, bench_world, bench_rf, dotcom_log_ret = load_data()

# ==========================================
# 3. RISK & MATH FUNCTIONS
# ==========================================
def calculate_historical_risk(log_returns, capital, var_level, days):
    if days == 1:
        rolling_discrete = np.exp(log_returns.dropna()) - 1
    else:
        rolling_log = log_returns.rolling(window=days).sum().dropna()
        if len(rolling_log) < 100:
            return np.nan, np.nan
        rolling_discrete = np.exp(rolling_log) - 1

    var_pct = np.percentile(rolling_discrete, var_level * 100)
    tail_returns = rolling_discrete[rolling_discrete <= var_pct]
    es_pct = np.mean(tail_returns)
    return capital * var_pct, capital * es_pct

def calculate_gaussian_risk(returns, capital, var_level, days):
    clean_returns = returns.dropna()
    mu_1d = np.mean(clean_returns)
    sigma_1d = np.std(clean_returns)
    mu_nd = mu_1d * days
    sigma_nd = sigma_1d * np.sqrt(days)
    z_score = stats.norm.ppf(var_level)
    var_pct = mu_nd + (z_score * sigma_nd)
    es_pct = mu_nd - sigma_nd * (stats.norm.pdf(z_score) / var_level)
    return capital * var_pct, capital * es_pct

def calculate_lognormal_risk(log_returns, capital, var_level, days):
    clean_returns = log_returns.dropna()
    mu_log_1d = np.mean(clean_returns)
    sigma_log_1d = np.std(clean_returns)
    mu_log_nd = mu_log_1d * days
    sigma_log_nd = sigma_log_1d * np.sqrt(days)
    z_score = stats.norm.ppf(var_level)
    log_worst_case = mu_log_nd + (z_score * sigma_log_nd)
    var_pct = np.exp(log_worst_case) - 1
    expected_value_lognorm = np.exp(mu_log_nd + (sigma_log_nd**2) / 2)
    tail_factor = stats.norm.cdf(z_score - sigma_log_nd) / var_level
    es_pct = (expected_value_lognorm * tail_factor) - 1
    return capital * var_pct, capital * es_pct

def calculate_monte_carlo_risk(log_returns, capital, var_level, days, simulations=10000, black_swan=False, crash_data=None):
    clean_returns = log_returns.dropna()
    mu_log_1d = np.mean(clean_returns)
    sigma_log_1d = np.std(clean_returns)
    np.random.seed(monte_carlo_seed)
    simulated_daily_returns = np.random.normal(mu_log_1d, sigma_log_1d, (days, simulations))

    if black_swan and crash_data is not None:
        crash_data_flat = np.asarray(crash_data).flatten()
        crash_length = len(crash_data_flat)
        if days < crash_length:
            slice_starts = np.random.randint(0, crash_length - days + 1, size=simulations)
            for s in range(simulations):
                start_idx_crash = slice_starts[s]
                simulated_daily_returns[:, s] = crash_data_flat[start_idx_crash : start_idx_crash + days]
        else:
            crash_start_days = np.random.randint(0, max(1, days - int(crash_length/2)), size=simulations)
            for s in range(simulations):
                start_idx = crash_start_days[s]
                end_idx = min(days, start_idx + crash_length)
                actual_crash_len = end_idx - start_idx
                simulated_daily_returns[start_idx:end_idx, s] = crash_data_flat[:actual_crash_len]

    cumulative_log_returns = np.cumsum(simulated_daily_returns, axis=0)
    portfolio_paths = capital * np.exp(cumulative_log_returns)
    final_values = portfolio_paths[-1, :]
    var_end_value = np.percentile(final_values, var_level * 100)
    tail_values = final_values[final_values <= var_end_value]
    es_end_value = np.mean(tail_values) if len(tail_values) > 0 else var_end_value

    var_PnL = var_end_value - capital
    es_PnL = es_end_value - capital
    return var_PnL, es_PnL, final_values, portfolio_paths

def calculate_performance_kpis(portfolio_returns, benchmark_world_data, benchmark_risk_free_data):
    benchmark_returns = benchmark_world_data.pct_change().dropna().squeeze()
    risk_free_daily = (benchmark_risk_free_data.dropna() / 100 / 252).squeeze()
    
    aligned_kpi_data = pd.concat([
        portfolio_returns.squeeze().rename('Portfolio'), 
        benchmark_returns.rename('Market'), 
        risk_free_daily.rename('RiskFree')
    ], axis=1, sort=False).dropna()

    port_ret = aligned_kpi_data['Portfolio']
    mkt_ret = aligned_kpi_data['Market']
    rf_ret = aligned_kpi_data['RiskFree']

    cov_matrix = np.cov(port_ret, mkt_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    mu_rp_daily = np.mean(port_ret)
    mu_rm_daily = np.mean(mkt_ret)
    mu_rf_daily = np.mean(rf_ret)
    sigma_rp_daily = np.std(port_ret)

    sharpe_ann = ((mu_rp_daily - mu_rf_daily) / sigma_rp_daily) * np.sqrt(252)
    rsf_ann = ((mu_rp_daily - mu_rm_daily) / sigma_rp_daily) * np.sqrt(252)
    treynor_ann = ((mu_rp_daily * 252) - (mu_rf_daily * 252)) / beta

    return {"Beta": beta, "Sharpe_Ratio": sharpe_ann, "Roys_Safety_First": rsf_ann, "Treynor_Ratio": treynor_ann}

def get_comparison_data(log_ret, discrete_ret, capital):
    results = []
    # Wir nehmen die zwei Standard-Level für den Subplot-Vergleich
    levels = {"95 %": 0.05, "99 %": 0.01}
    
    for h_name, days in horizons.items():
        for lvl_name, alpha in levels.items():
            # Berechne alle Methoden
            h_var, _ = calculate_historical_risk(log_ret, capital, alpha, days)
            g_var, _ = calculate_gaussian_risk(discrete_ret, capital, alpha, days)
            l_var, _ = calculate_lognormal_risk(log_ret, capital, alpha, days)
            
            # Daten für DataFrame sammeln
            results.append({"Horizont": h_name, "Konfidenz": lvl_name, "Methode": "Historisch", "VaR ($)": abs(h_var)})
            results.append({"Horizont": h_name, "Konfidenz": lvl_name, "Methode": "Gaußsch", "VaR ($)": abs(g_var)})
            results.append({"Horizont": h_name, "Konfidenz": lvl_name, "Methode": "Lognormal (MC)", "VaR ($)": abs(l_var)})
            
    return pd.DataFrame(results)
# ==========================================
# 4. PLOTTING FUNCTIONS
# ==========================================
def plot_monte_carlo_fan_chart(portfolio_paths, start_cap, var_level=0.05, title="Monte Carlo Fan-Chart"):
    days = portfolio_paths.shape[0]
    lower_pct, upper_pct = var_level * 100, (1 - var_level) * 100
    
    median_path = np.insert(np.median(portfolio_paths, axis=1), 0, start_cap)
    lower_bound = np.insert(np.percentile(portfolio_paths, lower_pct, axis=1), 0, start_cap)
    upper_bound = np.insert(np.percentile(portfolio_paths, upper_pct, axis=1), 0, start_cap)
    x_axis = np.insert(np.arange(1, days + 1), 0, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=lower_bound, mode='lines', line=dict(color='rgba(231, 76, 60, 0.8)', width=1), name=f'Worst Case ({lower_pct:g}%)'))
    fig.add_trace(go.Scatter(x=x_axis, y=median_path, mode='lines', line=dict(color='rgb(52, 152, 219)', width=3), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)', name='Median Pfad'))
    fig.add_trace(go.Scatter(x=x_axis, y=upper_bound, mode='lines', line=dict(color='rgba(46, 204, 113, 0.8)', width=1), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)', name=f'Best Case ({upper_pct:g}%)'))
    
    fig.add_hline(y=start_cap, line_dash="dash", line_color="rgba(255, 255, 255, 0.5)", annotation_text="Startkapital")
    fig.update_layout(title=title, xaxis_title="Handelstage", yaxis_title="Portfolio-Wert ($)", template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def plot_historical_performance(portfolio_returns, market_returns, risk_free_returns, start_capital=100000, title="Historische Renditeentwicklung"):
    aligned_data = pd.concat([
        portfolio_returns.squeeze().rename('Portfolio'), 
        market_returns.squeeze().rename('Market'), 
        risk_free_returns.squeeze().rename('RiskFree')
    ], axis=1, sort=False).dropna()
    portfolio_growth = start_capital * (1 + aligned_data['Portfolio']).cumprod()
    market_growth = start_capital * (1 + aligned_data['Market']).cumprod()
    risk_free_growth = start_capital * (1 + aligned_data['RiskFree']).cumprod()

    portfolio_growth.loc[aligned_data.index[0] - pd.Timedelta(days=1)] = start_capital
    market_growth.loc[aligned_data.index[0] - pd.Timedelta(days=1)] = start_capital
    risk_free_growth.loc[aligned_data.index[0] - pd.Timedelta(days=1)] = start_capital

    portfolio_growth = portfolio_growth.sort_index()
    market_growth = market_growth.sort_index()
    risk_free_growth = risk_free_growth.sort_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_growth.index, y=portfolio_growth,
        mode='lines', line=dict(color='rgb(52, 152, 219)', width=2), # Blau
        name='MAG7 Portfolio'
    ))

    fig.add_trace(go.Scatter(
        x=market_growth.index, y=market_growth,
        mode='lines', line=dict(color='rgb(231, 76, 60)', width=2), # Rot
        name='Markt (Benchmark)'
    ))

    fig.add_trace(go.Scatter(
        x=risk_free_growth.index, y=risk_free_growth,
        mode='lines', line=dict(color='rgb(46, 204, 113)', width=2, dash='dot'), # Grün gepunktet
        name='Risk-Free Rate'
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Datum",
        yaxis_title="Portfoliowert ($)",
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def plot_black_swan_comparison(paths_normal, paths_swan, start_cap, title="Median-Vergleich: Normal vs. Black Swan"):
    days = paths_normal.shape[0]
    median_normal = np.insert(np.median(paths_normal, axis=1), 0, start_cap)
    median_swan = np.insert(np.median(paths_swan, axis=1), 0, start_cap)
    x_axis = np.insert(np.arange(1, days + 1), 0, 0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=median_normal, mode='lines', line=dict(color='rgb(52, 152, 219)', width=3), name='Median (Ohne Crash)'))
    fig.add_trace(go.Scatter(x=x_axis, y=median_swan, mode='lines', line=dict(color='rgb(231, 76, 60)', width=3, dash='dash'), name='Median (Mit DotCom-Crash)'))
    fig.add_hline(y=start_cap, line_dash="dot", line_color="rgba(255, 255, 255, 0.3)", annotation_text="Startkapital")
    
    fig.update_layout(title=title, xaxis_title="Handelstage", yaxis_title="Portfolio-Wert ($)", template="plotly_dark", hovermode="x unified", margin=dict(l=20, r=20, t=50, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# ==========================================
# 5. UI HELPER FUNCTION
# ==========================================
def render_risk_tab(days, tab_title):
    st.header(tab_title)
    
    # Layout für A und B Selektion
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Auswahl VaR-Level - A")
        lvl_a_name = st.selectbox("VaR-Level A", list(var_levels_ui.keys()), key=f"sel_a_{days}", label_visibility="collapsed")
        alpha_a = var_levels_ui[lvl_a_name]
        
        # Berechnungen A
        h_var_a, h_es_a = calculate_historical_risk(port_ret_log, start_capital, alpha_a, days)
        g_var_a, g_es_a = calculate_gaussian_risk(port_ret_discrete, start_capital, alpha_a, days)
        l_var_a, l_es_a = calculate_lognormal_risk(port_ret_log, start_capital, alpha_a, days)
        
        st.write("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Historisch VaR", f"${h_var_a:,.0f}")
        c1.metric("Historisch ES", f"${h_es_a:,.0f}")
        c2.metric("Gaußsch VaR", f"${g_var_a:,.0f}")
        c2.metric("Gaußsch ES", f"${g_es_a:,.0f}")
        c3.metric("Lognormal VaR", f"${l_var_a:,.0f}")
        c3.metric("Lognormal ES", f"${l_es_a:,.0f}")

    with col_b:
        st.subheader("Auswahl VaR-Level - B")
        lvl_b_name = st.selectbox("VaR-Level B", list(var_levels_ui.keys()), key=f"sel_b_{days}", index=1, label_visibility="collapsed")
        alpha_b = var_levels_ui[lvl_b_name]
        
        # Berechnungen B
        h_var_b, h_es_b = calculate_historical_risk(port_ret_log, start_capital, alpha_b, days)
        g_var_b, g_es_b = calculate_gaussian_risk(port_ret_discrete, start_capital, alpha_b, days)
        l_var_b, l_es_b = calculate_lognormal_risk(port_ret_log, start_capital, alpha_b, days)
        
        st.write("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Historisch VaR", f"${h_var_b:,.0f}")
        c1.metric("Historisch ES", f"${h_es_b:,.0f}")
        c2.metric("Gaußsch VaR", f"${g_var_b:,.0f}")
        c2.metric("Gaußsch ES", f"${g_es_b:,.0f}")
        c3.metric("Lognormal VaR", f"${l_var_b:,.0f}")
        c3.metric("Lognormal ES", f"${l_es_b:,.0f}")

    # Visualisierungen A und B
    st.write("---")
    col_chart_a, col_chart_b = st.columns(2)
    
    # Monte Carlo Sim für Charts
    _, _, _, paths_a = calculate_monte_carlo_risk(port_ret_log, start_capital, alpha_a, days, simulations=2000)
    _, _, _, paths_b = calculate_monte_carlo_risk(port_ret_log, start_capital, alpha_b, days, simulations=2000)
    
    with col_chart_a:
        st.plotly_chart(plot_monte_carlo_fan_chart(paths_a, start_capital, alpha_a, f"Visualisierung VaR A ({lvl_a_name})"), use_container_width=True)
    with col_chart_b:
        st.plotly_chart(plot_monte_carlo_fan_chart(paths_b, start_capital, alpha_b, f"Visualisierung VaR B ({lvl_b_name})"), use_container_width=True)

# ==========================================
# 6. STREAMLIT APP LAYOUT
# ==========================================

# Sidebar
st.sidebar.title("Magnificent 7: Risiko - Dashboard")
start_capital = st.sidebar.number_input("Startkapital ($)", value=100_000, step=10_000)

# Tabs
tab_uebersicht, tab_1y, tab_5y, tab_10y, tab_black_swan = st.tabs([
    "Übersicht", "1-Jahres-Risiko", "5-Jahres-Risiko", "10-Jahres-Risiko", "Black-Swan-Sim"
])

# ----------------- REITER 0: ÜBERSICHT -----------------
with tab_uebersicht:
    st.header("Performance KPIs")
    kpis = calculate_performance_kpis(port_ret_discrete, bench_world, bench_rf)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beta", f"{kpis['Beta']:.2f}", help="Maß für die Schwankung des Portfolios im Vergleich zum Markt.")
    col2.metric("Sharpe Ratio", f"{kpis['Sharpe_Ratio']:.2f}", help="Überrendite pro Einheit Risiko (Volatilität).")
    col3.metric("Roy's Safety First", f"{kpis['Roys_Safety_First']:.2f}", help="Wahrscheinlichkeit, dass die Rendite unter eine Mindestrendite fällt.")
    col4.metric("Treynor-Ratio", f"{kpis['Treynor_Ratio']:.4f}", help="Überrendite pro Einheit des systematischen Risikos (Beta).")
    
    st.write("---")
    st.plotly_chart(plot_historical_performance(port_ret_discrete, bench_world, bench_rf, start_capital), use_container_width=True)

# ----------------- REITER 0: ÜBERSICHT -----------------
with tab_uebersicht:
    st.header("Performance KPIs")
    kpis = calculate_performance_kpis(port_ret_discrete, bench_world, bench_rf)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Beta", f"{kpis['Beta']:.2f}", help="Maß für die Schwankung des Portfolios im Vergleich zum Markt.")
    col2.metric("Sharpe Ratio", f"{kpis['Sharpe_Ratio']:.2f}", help="Überrendite pro Einheit Risiko (Volatilität).")
    col3.metric("Roy's Safety First", f"{kpis['Roys_Safety_First']:.2f}", help="Wahrscheinlichkeit, dass die Rendite unter eine Mindestrendite fällt.")
    col4.metric("Treynor-Ratio", f"{kpis['Treynor_Ratio']:.4f}", help="Überrendite pro Einheit des systematischen Risikos (Beta).")
    
    st.write("---")
   
    mkt_returns = bench_world.pct_change().dropna()
    rf_returns = bench_rf.dropna() / 100 / 252 
    fig_hist = plot_historical_performance(
     portfolio_returns = port_ret_discrete, 
     market_returns = mkt_returns, 
     risk_free_returns = rf_returns, 
     start_capital = start_capital,
     title = "Performance MAG7 vs. Markt vs. Risikofreier Zins (2012-2026)"
 )
    st.plotly_chart(fig_hist, use_container_width=True)

# ----------------- REITER 1, 2, 3: RISIKO-HORIZONTE -----------------
with tab_1y:
    render_risk_tab(horizons["1 Jahr"], "1-Jahres-Risiko")
    
with tab_5y:
    render_risk_tab(horizons["5 Jahre"], "5-Jahres-Risiko")

with tab_10y:
    render_risk_tab(horizons["10 Jahre"], "10-Jahres-Risiko")

# ----------------- REITER 4: BLACK SWAN -----------------
with tab_black_swan:
    st.header("Black-Swan-Simulation")
    st.info("Das Black-Swan-Event simuliert den VaR & ES auf Basis eines im Zeitverlauf zufällig eintretenden Black-Swan. Dieses Event wurde mit den Renditen & Volatilität aus dem Crash der Dotcom-Blase simuliert.")
    
    col1, col2 = st.columns(2)
    with col1:
        years_swan = st.selectbox("Auswahl Anzahl Jahre", [1, 5, 10, 20])
        days_swan = horizons[f"{years_swan} Jahr" if years_swan == 1 else f"{years_swan} Jahre"]
    with col2:
        swan_lvl_name = st.selectbox("Auswahl VaR-Level", list(var_levels_ui.keys()), key="var_swan")
        alpha_swan = var_levels_ui[swan_lvl_name]
    
    # Berechnungen Normal vs Swan
    mc_var_norm, mc_es_norm, _, paths_norm = calculate_monte_carlo_risk(
        port_ret_log, start_capital, alpha_swan, days_swan, simulations=2000, black_swan=False
    )
    mc_var_swan, mc_es_swan, _, paths_swan = calculate_monte_carlo_risk(
        port_ret_log, start_capital, alpha_swan, days_swan, simulations=2000, black_swan=True, crash_data=dotcom_log_ret
    )
    
    st.write("---")
    c1, c2 = st.columns(2)
    c1.subheader("Ohne Blackswan ('normal')")
    c1.metric("VaR ($)", f"${mc_var_norm:,.0f}")
    c1.metric("ES ($)", f"${mc_es_norm:,.0f}")
    
    c2.subheader("Mit Blackswan")
    c2.metric("VaR ($)", f"${mc_var_swan:,.0f}")
    c2.metric("ES ($)", f"${mc_es_swan:,.0f}")

    st.write("---")
    st.plotly_chart(plot_black_swan_comparison(paths_norm, paths_swan, start_capital), use_container_width=True)
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.plotly_chart(plot_monte_carlo_fan_chart(paths_norm, start_capital, alpha_swan, "Visualisierung Ohne Blackswan"), use_container_width=True)
    with col_chart2:
        st.plotly_chart(plot_monte_carlo_fan_chart(paths_swan, start_capital, alpha_swan, "Visualisierung Black Swan"), use_container_width=True)