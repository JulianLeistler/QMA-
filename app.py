import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURATION & DATEN LADEN
# ==========================================
st.set_page_config(page_title="Mag7 Risk Dashboard", layout="wide")

@st.cache_data
def get_portfolio_data():
    tickers = ['AAPL', 'TSLA', 'MSFT', 'META', 'AMZN', 'GOOGL', 'NVDA']
    # Download
    data = yf.download(tickers, start="2016-04-01", end="2026-04-01", auto_adjust=False)['Close']
    
    # Berechnung der täglichen Log-Renditen (Gleichgewichtet)
    daily_simple_returns = data.pct_change().mean(axis=1)
    portfolio_log_returns = np.log(1 + daily_simple_returns).dropna()
    
    return portfolio_log_returns

# Daten abrufen
try:
    log_returns = get_portfolio_data()
except Exception as e:
    st.error(f"Fehler beim Datendownload: {e}")
    st.stop()

# ==========================================
# 2. BERECHNUNGS-FUNKTIONEN (BACKEND)
# ==========================================

def calculate_monte_carlo_risk(log_returns, capital, var_level, days, simulations=10000, black_swan=False, inflation_rate=0.0):
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)
    np.random.seed(2)
    sim_returns = np.random.normal(mu, sigma, (days, simulations))
    final_values = capital * np.exp(np.sum(sim_returns, axis=0))
    
    if inflation_rate > 0.0:
        final_values /= (1 + inflation_rate)**(days/252)
    if black_swan:
        final_values *= 0.5
        
    var_val = np.percentile(final_values, (1 - var_level) * 100)
    es_val = np.mean(final_values[final_values <= var_val])
    return var_val - capital, es_val - capital

def calculate_bootstrap_risk(log_returns, capital, var_level, days, simulations=10000, black_swan=False, inflation_rate=0.0):
    np.random.seed(1)
    sim_returns = np.random.choice(log_returns, size=(days, simulations), replace=True)
    final_values = capital * np.exp(np.sum(sim_returns, axis=0))
    
    if inflation_rate > 0.0:
        final_values /= (1 + inflation_rate)**(days/252)
    if black_swan:
        final_values *= 0.5
        
    var_val = np.percentile(final_values, (1 - var_level) * 100)
    es_val = np.mean(final_values[final_values <= var_val])
    return var_val - capital, es_val - capital

def calculate_savings_plan_mc(log_returns, monthly_savings, years, simulations=10000, black_swan=False, inflation_rate=0.0):
    days = years * 252
    mu, sigma = np.mean(log_returns), np.std(log_returns)
    np.random.seed(2)
    daily_sim_returns = np.exp(np.random.normal(mu, sigma, (days, simulations)))
    
    if black_swan:
        crash_days = np.random.randint(0, days, size=simulations)
        for s in range(simulations):
            daily_sim_returns[crash_days[s], s] *= 0.5

    savings_matrix = np.zeros((days, simulations))
    for d in range(0, days, 21):
        savings_matrix[d, :] = monthly_savings
    
    portfolio_values = np.zeros((days, simulations))
    current_values = np.zeros(simulations)
    for d in range(days):
        current_values = current_values * daily_sim_returns[d, :] + savings_matrix[d, :]
        portfolio_values[d, :] = current_values
        
    final_values = portfolio_values[-1, :]
    if inflation_rate > 0.0:
        final_values /= (1 + inflation_rate)**years
        
    total_invested = (days // 21) * monthly_savings
    var_val = np.percentile(final_values, 5)
    es_val = np.mean(final_values[final_values <= var_val])
    
    return var_val - total_invested, es_val - total_invested, final_values, total_invested, portfolio_values

# ==========================================
# 3. UI LAYOUT (FRONTEND)
# ==========================================

st.title("🚀 Magnificent 7 - Portfolio Risiko Dashboard")
st.markdown("Analyse der Mag7-Aktien mittels Monte Carlo, Bootstrapping und Sparplan-Simulationen.")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Globale Parameter")
capital = st.sidebar.number_input("Anfangskapital ($)", value=100000, step=1000)
var_level = st.sidebar.slider("Konfidenzniveau (VaR)", 0.90, 0.99, 0.95, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🌪️ Stresstests & Kaufkraft")
black_swan_active = st.sidebar.toggle("Black Swan Modus (-50% Crash)")
inflation = st.sidebar.slider("Inflation p.a. (%)", 0.0, 10.0, 2.0) / 100

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Historie", "💰 Einmalanlage", "📈 Sparplan"])

with tab1:
    st.header("Historische Performance")
    st.info("Hier kannst du deine bereits berechneten Performance-Metriken einfügen.")
    # Platzhalter für deine Metriken
    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", "MAG7 Portfolio")
    col2.metric("Zeitraum", "10 Jahre")
    col3.metric("Status", "Bereit")

with tab2:
    st.header("Risiko-Analyse (Einmalanlage)")
    horizon = st.selectbox("Anlagehorizont", [1, 5, 10], index=1)
    days_horizon = horizon * 252
    
    # Berechnungen
    var_mc, es_mc = calculate_monte_carlo_risk(log_returns, capital, var_level, days_horizon, black_swan=black_swan_active, inflation_rate=inflation)
    var_bs, es_bs = calculate_bootstrap_risk(log_returns, capital, var_level, days_horizon, black_swan=black_swan_active, inflation_rate=inflation)
    
    # Anzeige als Tabelle
    results = pd.DataFrame({
        "Methode": ["Monte Carlo (Lognormal)", "Bootstrapping (Empirisch)"],
        f"VaR {var_level*100:.0f}% ($)": [f"{var_mc:,.2f}", f"{var_bs:,.2f}"],
        f"Exp. Shortfall ($)": [f"{es_mc:,.2f}", f"{es_bs:,.2f}"]
    })
    st.table(results)
    st.caption(f"Beträge zeigen Gewinn/Verlust im Vergleich zum Startkapital von {capital:,} $")

with tab3:
    st.header("Zukunftssimulation (Sparplan)")
    col_a, col_b = st.columns(2)
    savings_rate = col_a.number_input("Monatliche Rate ($)", value=500, step=50)
    savings_years = col_b.slider("Laufzeit (Jahre)", 5, 30, 20)
    
    # Berechnung
    v_pnl, e_pnl, finals, total_inv, p_matrix = calculate_savings_plan_mc(
        log_returns, savings_rate, savings_years, black_swan=black_swan_active, inflation_rate=inflation
    )
    
    # Plotly Chart
    fig = go.Figure()
    x_axis = np.arange(savings_years * 252) / 252
    # Zeichne 100 zufällige Pfade
    for i in np.random.choice(range(10000), 100):
        fig.add_trace(go.Scatter(x=x_axis, y=p_matrix[:, i], mode='lines', 
                                 line=dict(color='rgba(0, 150, 255, 0.1)'), showlegend=False))
    
    # Investiertes Kapital Linie
    invested_line = (np.arange(savings_years * 252) // 21) * savings_rate
    fig.add_trace(go.Scatter(x=x_axis, y=invested_line, name="Investiertes Kapital", line=dict(color='white', dash='dash')))
    
    fig.update_layout(template="plotly_dark", title="Simulation der Sparplan-Pfade", xaxis_title="Jahre", yaxis_title="Portfolio-Wert ($)")
    st.plotly_chart(fig, use_container_width=True)
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Gesamt investiert", f"{total_inv:,.0f} $")
    kpi2.metric("Durchschnittlicher Endwert", f"{np.mean(finals):,.2f} $")
    kpi3.metric("VaR (Worst 5%) PnL", f"{v_pnl:,.2f} $", delta_color="normal")