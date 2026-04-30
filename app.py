import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go

# --- SEITEN-KONFIGURATION ---
st.set_page_config(page_title="Mag7 Risk & Performance Dashboard", layout="wide")

# --- CACHE DATEN-LADEFUNKTION ---
@st.cache_data(ttl=86400) # Cacht die Daten für 24 Stunden
def load_data():
    tickers = ['AAPL', 'TSLA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'NVDA']
    data = yf.download(tickers, start="2016-04-01", end="2026-04-01", auto_adjust=False)['Close']
    benchmark_world_data = yf.download("VT", start="2016-04-01", end="2026-04-01", auto_adjust=False)['Close']
    benchmark_risk_free_data = yf.download("^TNX", start="2016-04-01", end="2026-04-01", auto_adjust=False)['Close']
    
    # Berechnung der täglichen Renditen
    returns_discrete = data.pct_change().dropna()
    returns_log = np.log(data / data.shift(1)).dropna()
    
    # Portfolio-Gewichtung (1/7)
    weights = np.array([1/len(tickers)] * len(tickers))
    
    portfolio_returns_discrete = returns_discrete.dot(weights)
    portfolio_returns_log = returns_log.dot(weights)
    
    return portfolio_returns_discrete, portfolio_returns_log, benchmark_world_data, benchmark_risk_free_data

# Daten laden
try:
    port_ret_disc, port_ret_log, bench_world, bench_rf = load_data()
except Exception as e:
    st.error("Fehler beim Laden der Finanzdaten. Bitte Internetverbindung prüfen.")
    st.stop()

# --- SIDEBAR (BENUTZEREINGABEN) ---
st.sidebar.header("⚙️ Parameter Einstellungen")

start_capital = st.sidebar.number_input("Startkapital ($)", min_value=1000, value=100000, step=5000)
var_level_pct = st.sidebar.slider("VaR Level (Alpha) in %", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
alpha_level = var_level_pct / 100.0

inflation_rate_pct = st.sidebar.slider("Inflationsrate (p.a.) in %", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
inflation_rate = inflation_rate_pct / 100.0

black_swan = st.sidebar.checkbox("Black-Swan Event simulieren (50% Crash)", value=False)

st.sidebar.divider()
st.sidebar.subheader("Sparplan Parameter")
monatliche_rate = st.sidebar.number_input("Monatliche Sparrate ($)", min_value=10, value=100, step=10)


# --- FUNKTIONEN (Aus deinem originalen Code) ---
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

@st.cache_data(show_spinner=False)
def run_monte_carlo(log_returns, capital, days, simulations=10000, black_swan=False, inflation_rate=0.0):
    clean_returns = log_returns.dropna()
    mu_log_1d = np.mean(clean_returns)
    sigma_log_1d = np.std(clean_returns)
    np.random.seed(2)
    simulated_daily_returns = np.random.normal(mu_log_1d, sigma_log_1d, (days, simulations))
    cumulative_log_returns = np.sum(simulated_daily_returns, axis=0)
    final_values = capital * np.exp(cumulative_log_returns)
    
    if inflation_rate > 0.0:
        discount_factor = (1 + inflation_rate) ** (days / 252)
        final_values = final_values / discount_factor
    if black_swan:
        final_values = final_values * 0.5
        
    # Für Plotly speichern wir auch die Pfad-Entwicklung (für die ersten 100 Pfade)
    path_evolution = capital * np.exp(np.cumsum(simulated_daily_returns[:, :100], axis=0))
    return final_values, path_evolution

@st.cache_data(show_spinner=False)
def run_savings_mc(log_returns, monthly_savings, years, simulations=10000, black_swan=False, inflation_rate=0.0):
    days = years * 252
    clean_returns = log_returns.dropna()
    mu = np.mean(clean_returns)
    sigma = np.std(clean_returns)
    np.random.seed(2)
    
    daily_log_returns = np.random.normal(mu, sigma, (days, simulations))
    daily_simple_returns = np.exp(daily_log_returns)
    
    if black_swan:
        crash_days = np.random.randint(0, days, size=simulations)
        for sim_idx in range(simulations):
            daily_simple_returns[crash_days[sim_idx], sim_idx] *= 0.5

    savings_matrix = np.zeros((days, simulations))
    for d in range(0, days, 21):
        savings_matrix[d, :] = monthly_savings
        
    current_values = np.zeros(simulations)
    portfolio_values_subset = np.zeros((days, 100)) # Nur für die Visualisierung (100 Pfade)
    
    for d in range(days):
        current_values = current_values * daily_simple_returns[d, :] + savings_matrix[d, :]
        if d < days:
             portfolio_values_subset[d, :] = current_values[:100]
             
    final_values = current_values
    if inflation_rate > 0.0:
        final_values = final_values / ((1 + inflation_rate) ** years)
        
    total_invested = (days // 21) * monthly_savings
    return final_values, total_invested, portfolio_values_subset

# --- MAIN DASHBOARD ---
st.title("📈 Magnificent 7: Risiko & Portfolio Dashboard")
st.markdown("Analyse des gleichgewichteten Portfolios der Mag7 (Apple, Tesla, Microsoft, Meta, Amazon, Alphabet, NVIDIA).")

tab1, tab2, tab3 = st.tabs(["📊 1-Jahres-Risiko & KPIs", "🔮 Langfristiges Risiko (MC)", "💰 Sparplan-Simulation"])

# === TAB 1: 1 JAHRES RISIKO & KPIs ===
with tab1:
    st.header("1-Jahres-Risiko (252 Handelstage)")
    col1, col2, col3 = st.columns(3)
    
    h_var, h_es = calculate_historical_risk(port_ret_log, start_capital, alpha_level, 252)
    g_var, g_es = calculate_gaussian_risk(port_ret_disc, start_capital, alpha_level, 252)
    l_var, l_es = calculate_lognormal_risk(port_ret_log, start_capital, alpha_level, 252)
    
    col1.metric("Historisch (BHS) - VaR", f"$ {abs(h_var):,.2f}", f"ES: $ {abs(h_es):,.2f}", delta_color="inverse")
    col2.metric("Gaußsch (Normal) - VaR", f"$ {abs(g_var):,.2f}", f"ES: $ {abs(g_es):,.2f}", delta_color="inverse")
    col3.metric("Lognormal - VaR", f"$ {abs(l_var):,.2f}", f"ES: $ {abs(l_es):,.2f}", delta_color="inverse")

    st.divider()
    
    st.subheader("Performance KPIs (Annualisiert)")
    kpis = calculate_performance_kpis(port_ret_disc, bench_world, bench_rf)
    
    kcol1, kcol2, kcol3, kcol4 = st.columns(4)
    kcol1.metric("Beta", f"{kpis['Beta']:.2f}")
    kcol2.metric("Sharpe Ratio", f"{kpis['Sharpe_Ratio']:.2f}")
    kcol3.metric("Roy's Safety First", f"{kpis['Roys_Safety_First']:.2f}")
    kcol4.metric("Treynor Ratio", f"{kpis['Treynor_Ratio']:.4f}")
    
    with st.expander("ℹ️ Definitionen der Performance KPIs"):
        st.markdown("""
        * **Beta:** Maß für die Schwankungsanfälligkeit des Portfolios im Vergleich zum Gesamtmarkt (VT ETF). > 1 bedeutet riskanter als der Markt.
        * **Sharpe Ratio:** Misst die Überrendite pro Einheit des Gesamtrisikos (Volatilität). Höher ist besser.
        * **Roy's Safety First:** Misst die Wahrscheinlichkeit, dass die Rendite unter eine definierte Mindestrendite (hier: Marktrendite) fällt.
        * **Treynor Ratio:** Misst die Überrendite pro Einheit des systematischen Risikos (Beta).
        """)
        
    st.divider()
    st.subheader("📈 Vergleichschart: Mag7 vs. Weltmarkt (VT)")
    # Kumulierte Rendite berechnen
    mag7_cum = (1 + port_ret_disc).cumprod() * 100
    vt_ret_disc = bench_world.pct_change().dropna()
    vt_cum = (1 + vt_ret_disc).cumprod() * 100
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=mag7_cum.index, y=mag7_cum, mode='lines', name='Mag7 Portfolio', line=dict(color='blue')))
    fig_comp.add_trace(go.Scatter(x=vt_cum.index, y=vt_cum, mode='lines', name='Vanguard World (VT)', line=dict(color='gray')))
    fig_comp.update_layout(yaxis_title="Wertentwicklung (Basis 100)", xaxis_title="Datum", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_comp, use_container_width=True)

# === TAB 2: LANGFRISTIGES RISIKO ===
with tab2:
    st.header("Monte-Carlo-Simulation (Langfristig)")
    horizon_label = st.selectbox("Anlagehorizont wählen:", ["5 Jahre", "10 Jahre"])
    days = 1260 if horizon_label == "5 Jahre" else 2520
    
    with st.spinner('Berechne 10.000 Monte Carlo Pfade...'):
        final_vals, paths = run_monte_carlo(port_ret_log, start_capital, days, 10000, black_swan, inflation_rate)
    
    var_end_value = np.percentile(final_vals, alpha_level * 100)
    tail_values = final_vals[final_vals <= var_end_value]
    es_end_value = np.mean(tail_values)
    
    mc_var_pnl = var_end_value - start_capital
    mc_es_pnl = es_end_value - start_capital

    col1, col2, col3 = st.columns(3)
    col1.metric("Durchschnittlicher Endwert", f"$ {np.mean(final_vals):,.2f}")
    col2.metric(f"Value at Risk ({var_level_pct}%)", f"$ {mc_var_pnl:,.2f}", "Verlust bei Eintreten", delta_color="inverse")
    col3.metric("Expected Shortfall", f"$ {mc_es_pnl:,.2f}", "Durchschn. Verlust im Tail", delta_color="inverse")

    st.subheader("Simulierte Portfolio-Entwicklung (Beispielhafte 100 Pfade)")
    fig_mc = go.Figure()
    # Wir zeichnen die ersten 100 Pfade
    x_axis = np.arange(days)
    for i in range(100):
        fig_mc.add_trace(go.Scatter(x=x_axis, y=paths[:, i], mode='lines', line=dict(width=1, color='rgba(0, 100, 250, 0.1)'), showlegend=False))
    
    # Durchschnittslinie hinzufügen
    fig_mc.add_trace(go.Scatter(x=x_axis, y=np.mean(paths, axis=1), mode='lines', name='Durchschnitt', line=dict(color='red', width=3)))
    fig_mc.update_layout(yaxis_title="Portfoliowert in $", xaxis_title="Handelstage", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_mc, use_container_width=True)


# === TAB 3: SPARPLAN ===
with tab3:
    st.header("Sparplan Monte-Carlo-Simulation")
    sp_horizon = st.selectbox("Sparplan-Laufzeit:", ["10 Jahre", "15 Jahre", "20 Jahre", "30 Jahre"], index=2)
    sp_years = int(sp_horizon.split()[0])
    
    with st.spinner(f'Berechne Sparplan für {sp_years} Jahre...'):
        sp_final_vals, sp_total_invested, sp_paths = run_savings_mc(
            port_ret_log, monatliche_rate, sp_years, 10000, black_swan, inflation_rate
        )
        
    sp_var_end_value = np.percentile(sp_final_vals, alpha_level * 100)
    sp_es_end_value = np.mean(sp_final_vals[sp_final_vals <= sp_var_end_value])
    
    sp_var_pnl = sp_var_end_value - sp_total_invested
    sp_es_pnl = sp_es_end_value - sp_total_invested

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gesamt Investiert", f"$ {sp_total_invested:,.2f}")
    col2.metric("Ø Endwert", f"$ {np.mean(sp_final_vals):,.2f}")
    col3.metric(f"VaR ({var_level_pct}%) PnL", f"$ {sp_var_pnl:,.2f}", "Gegenüber Einzahlung")
    col4.metric("ES PnL", f"$ {sp_es_pnl:,.2f}", "Gegenüber Einzahlung")

    st.subheader("Sparplan Vermögensaufbau (Beispielhafte 100 Pfade)")
    fig_sp = go.Figure()
    sp_x_axis = np.arange(sp_years * 252)
    
    # 100 Pfade plotten
    for i in range(100):
        fig_sp.add_trace(go.Scatter(x=sp_x_axis, y=sp_paths[:, i], mode='lines', line=dict(width=1, color='rgba(34, 139, 34, 0.1)'), showlegend=False))
        
    # Einzahlungssumme als Referenzlinie
    invested_line = np.linspace(0, sp_total_invested, sp_years * 252)
    fig_sp.add_trace(go.Scatter(x=sp_x_axis, y=invested_line, mode='lines', name='Eingezahltes Kapital', line=dict(color='black', width=3, dash='dash')))
    
    fig_sp.update_layout(yaxis_title="Vermögen in $", xaxis_title="Handelstage", template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_sp, use_container_width=True)