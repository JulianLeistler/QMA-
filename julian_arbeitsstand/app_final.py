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
benchmark_sp500_ticker = '^SPX'
benchmark_nasdaq_ticker = '^NDX'
start_date = "2012-05-18"
end_date = "2026-04-01"

horizons = {
    "1 Jahr": 252,
    "5 Jahre": 1260,
    "10 Jahre": 2520,
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
    bench_sp500 = yf.download(benchmark_sp500_ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
    bench_nasdaq = yf.download(benchmark_nasdaq_ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)['Adj Close']
    dotcom = yf.download('^NDX', start="2000-04-01", end="2002-10-31", auto_adjust=False, progress=False)['Adj Close']
    
    # Berechnungen
    dotcom_log_ret = np.log(dotcom / dotcom.shift(1)).dropna().values
    returns_discrete = data.pct_change().dropna()
    returns_log = np.log(data / data.shift(1)).dropna()
    weights = np.array([1/len(tickers)] * len(tickers))
    port_ret_discrete = returns_discrete.dot(weights)
    port_ret_log = returns_log.dot(weights)
    
    return port_ret_discrete, port_ret_log, bench_world, bench_rf, bench_sp500, bench_nasdaq, dotcom_log_ret

port_ret_discrete, port_ret_log, bench_world, bench_rf, bench_sp500, bench_nasdaq, dotcom_log_ret = load_data()

# ==========================================
# 3. RISK & MATH FUNCTIONS
# ==========================================
def calculate_historical_risk(log_returns, capital, var_level, days):
    if days == 1:
        rolling_discrete = np.exp(log_returns.dropna()) - 1 #keine Aggregation, da nur 1 Tag betrachtet wird
    else:
        rolling_log = log_returns.rolling(window=days).sum().dropna() # Rolling Window und Aggregation

        if len(rolling_log) < 100: #Sicherheitscheck: nicht weniger als 100 Beobachtungen für die Berechnung
            return np.nan, np.nan
        rolling_discrete = np.exp(rolling_log) - 1 #Log zu diskret

    var_pct = np.percentile(rolling_discrete, var_level * 100) 
    tail_returns = rolling_discrete[rolling_discrete <= var_pct]
    es_pct = np.mean(tail_returns)

    var_end_value = capital * (1 + var_pct)
    es_end_value = capital * (1 + es_pct)

    # 3. PnL-Logik (Endwert - Startkapital)
    var_pnl = var_end_value - capital
    es_pnl = es_end_value - capital

    return var_pnl, es_pnl

def calculate_gaussian_risk(returns, capital, var_level, days):
    clean_returns = returns.dropna()

    mu_1d = np.mean(clean_returns) #Erwartungswert der täglichen Renditen
    sigma_1d = np.std(clean_returns) #Volatilität der täglichen Renditen

    mu_nd = mu_1d * days 
    sigma_nd = sigma_1d * np.sqrt(days) #Skalierung der Volatilität mit der Quadratwurzel der Zeit

    z_score = stats.norm.ppf(var_level)

    var_pct = mu_nd + (z_score * sigma_nd)
    es_pct = mu_nd - sigma_nd * (stats.norm.pdf(z_score) / var_level)

    var_end_value = capital * (1 + var_pct)
    es_end_value = capital * (1 + es_pct)

    # 3. PnL-Logik (Endwert - Startkapital)
    var_pnl = var_end_value - capital
    es_pnl = es_end_value - capital

    return var_pnl, es_pnl

def calculate_lognormal_risk(log_returns, capital, var_level, days):
    clean_returns = log_returns.dropna()

    mu_log_1d = np.mean(clean_returns)
    sigma_log_1d = np.std(clean_returns)

    mu_log_nd = mu_log_1d * days
    sigma_log_nd = sigma_log_1d * np.sqrt(days)

    z_score = stats.norm.ppf(var_level)

    log_worst_case = mu_log_nd + (z_score * sigma_log_nd)
    var_end_value = capital * np.exp(log_worst_case)

    expected_value_lognorm = np.exp(mu_log_nd + (sigma_log_nd**2) / 2)
    tail_factor = stats.norm.cdf(z_score - sigma_log_nd) / var_level
    es_end_value = capital * expected_value_lognorm * tail_factor

    # 2. PnL-Logik (Endwert - Startkapital)
    var_pnl = var_end_value - capital
    es_pnl = es_end_value - capital

    return var_pnl, es_pnl

def calculate_bootstrap_risk(log_returns, capital, var_level, days, simulations=10000):
    clean_returns = log_returns.dropna()

    np.random.seed(bootstrap_seed)
    simulated_daily_returns = np.random.choice(clean_returns, size=(days, simulations), replace=True) #Bootstrap-Sampling der täglichen log Renditen
    cumulative_log_returns = np.sum(simulated_daily_returns, axis=0) 
    final_values = capital * np.exp(cumulative_log_returns)

    var_end_value = np.percentile(final_values, var_level * 100)
    tail_values = final_values[final_values <= var_end_value]
    es_end_value = np.mean(tail_values)

    var_PnL = var_end_value - capital
    es_PnL = es_end_value - capital

    return var_PnL, es_PnL

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
    # Standard-Level für den Subplot-Vergleich
    levels = {"95 %": 0.05, "99 %": 0.01}
    
    for h_name, days in horizons.items():
        for lvl_name, alpha in levels.items():
            
            # 1. Historische Ansätze (Parallel berechnet für den direkten Vergleich)
            bhs_var, _ = calculate_historical_risk(log_ret, capital, alpha, days)
            boot_var, _ = calculate_bootstrap_risk(log_ret, capital, alpha, days, simulations=10000)
            
            # 2. Parametrischer Ansatz
            g_var, _ = calculate_gaussian_risk(discrete_ret, capital, alpha, days)
            
            # 3. Lognormal Simulation (ohne Black Swan Event)
            mc_var, _, _, _ = calculate_monte_carlo_risk(
                log_ret, capital, alpha, days, 
                simulations=10000, black_swan=False
            )
            
            # Daten im exakten Format für die Plotly-Methode aggregieren
            results.extend([
                {"Horizont": h_name, "Konfidenz": lvl_name, "Methode": "Historisch (BHS)", "VaR ($)": bhs_var},
                {"Horizont": h_name, "Konfidenz": lvl_name, "Methode": "Historisch (Bootstrapping)", "VaR ($)": boot_var},
                {"Horizont": h_name, "Konfidenz": lvl_name, "Methode": "Gaußsch", "VaR ($)": g_var},
                {"Horizont": h_name, "Konfidenz": lvl_name, "Methode": "Lognormal (MC)", "VaR ($)": mc_var}
            ])
            
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

def plot_historical_performance(portfolio_returns, market_returns, sp500_returns, nasdaq_returns, start_capital=100000, title="Historische Renditeentwicklung"):
    aligned_data = pd.concat([
        portfolio_returns.squeeze().rename('Portfolio'), 
        market_returns.squeeze().rename('Market'), 
        sp500_returns.squeeze().rename('S&P 500'),
        nasdaq_returns.squeeze().rename('NASDAQ')
    ], axis=1, sort=False).dropna()

    portfolio_growth = start_capital * (1 + aligned_data['Portfolio']).cumprod()
    market_growth = start_capital * (1 + aligned_data['Market']).cumprod()
    sp500_growth = start_capital * (1 + aligned_data['S&P 500']).cumprod()
    nasdaq_growth = start_capital * (1 + aligned_data['NASDAQ']).cumprod()

    portfolio_growth.loc[aligned_data.index[0] - pd.Timedelta(days=1)] = start_capital
    market_growth.loc[aligned_data.index[0] - pd.Timedelta(days=1)] = start_capital
    sp500_growth.loc[aligned_data.index[0] - pd.Timedelta(days=1)] = start_capital
    nasdaq_growth.loc[aligned_data.index[0] - pd.Timedelta(days=1)] = start_capital

    portfolio_growth = portfolio_growth.sort_index()
    market_growth = market_growth.sort_index()
    sp500_growth = sp500_growth.sort_index()
    nasdaq_growth = nasdaq_growth.sort_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_growth.index, y=portfolio_growth,
        mode='lines', line=dict(color='rgb(52, 152, 219)', width=2), # Blau
        name='MAG7 Portfolio'
    ))

    fig.add_trace(go.Scatter(
        x=market_growth.index, y=market_growth,
        mode='lines', line=dict(color='rgb(231, 76, 60)', width=2), # Rot
        name='All World ETF'
    ))

    fig.add_trace(go.Scatter(
        x=sp500_growth.index, y=sp500_growth,
        mode='lines', line=dict(color='rgb(46, 204, 113)', width=2, dash='dot'), # Grün gepunktet
        name='S&P 500 ETF'
    ))

    fig.add_trace(go.Scatter(
        x=nasdaq_growth.index, y=nasdaq_growth,
        mode='lines', line=dict(color='rgb(142, 68, 173)', width=2, dash='dash'), # Lila gestrichelt
        name='NASDAQ 100 ETF'
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

def plot_var_bar_comparison(df_comparison, horizons, levels_to_show):
    plot_methods = [
        'Historisch (BHS)', 
        'Historisch (Bootstrapping)', 
        'Gaußsch', 
        'Lognormal (MC)'
    ]
    
    method_colors = {
        'Historisch (BHS)': 'rgb(133, 193, 233)',
        'Historisch (Bootstrapping)': 'rgb(41, 128, 185)',
        'Gaußsch': 'rgb(231, 76, 60)',
        'Lognormal (MC)': 'rgb(35, 155, 86)'
    }

    fig_p2 = make_subplots(
        rows=1, cols=len(levels_to_show), 
        subplot_titles=[f"{lvl} Konfidenz" for lvl in levels_to_show],
        shared_yaxes=True
    )

    horizon_order = list(horizons.keys())
    
    for col_idx, conf_label in enumerate(levels_to_show, start=1):
        subset = df_comparison[df_comparison['Konfidenz'] == conf_label]
        
        for method in plot_methods:
            method_data = subset[subset['Methode'] == method]
            if method_data.empty:
                continue
                
            method_data = method_data.set_index('Horizont').reindex(horizon_order).reset_index()
            
            fig_p2.add_trace(
                go.Bar(
                    x=method_data['Horizont'],
                    y=method_data['VaR ($)'],
                    name=method,
                    marker_color=method_colors.get(method, 'gray'),
                    showlegend=(col_idx == 1),
                    legendgroup=method 
                ),
                row=1, col=col_idx,
            )
            
        fig_p2.update_xaxes(title_text='Horizont', row=1, col=col_idx)

    fig_p2.update_yaxes(title_text='VaR (USD)', row=1, col=1)

    fig_p2.update_layout(
        template='plotly_dark',
        title='Methodenvergleich: Analytische Skalierung vs. Dynamische Simulation',
        barmode='group',
        height=550,
        margin=dict(t=80, b=20, l=20, r=20)
    )
    
    return fig_p2


def plot_density_comparison(portfolio_returns_log, portfolio_returns_discrete, start_capital, days, horizon_label, alpha):
    """Erstellt Plot C (Dichte) dynamisch basierend auf Horizont und VaR-Level."""
    
    # 1. Historisch (BHS): Rollierende P&L
    if days == 1:
        rolling_log_bhs = portfolio_returns_log.dropna()
    else:
        rolling_log_bhs = portfolio_returns_log.rolling(window=days).sum().dropna()
    
    hist_pnl_bhs = ((np.exp(rolling_log_bhs) - 1) * start_capital).values
    
    # 2. Historisch (Bootstrapping): Ziehen mit Zurücklegen
    simulations_boot = 2000
    bootstrapped_returns = np.random.choice(portfolio_returns_log.dropna(), size=(days, simulations_boot), replace=True)
    cumulative_log_returns = np.sum(bootstrapped_returns, axis=0)
    hist_pnl_boot = ((np.exp(cumulative_log_returns) - 1) * start_capital)

    # 3. Gauss: Analytische Dichte
    clean_disc = portfolio_returns_discrete.dropna()
    mu_g = clean_disc.mean() * days
    sigma_g = clean_disc.std(ddof=1) * np.sqrt(days)

    # 4. Lognormal MC
    _, _, final_values_mc, _ = calculate_monte_carlo_risk(
        portfolio_returns_log, start_capital, alpha, days,
        simulations=simulations_boot, black_swan=False,
    )
    mc_pnl = final_values_mc - start_capital

    # Gemeinsames x-Grid
    x_lo = float(min(hist_pnl_bhs.min() if len(hist_pnl_bhs) > 0 else 0, hist_pnl_boot.min(), mc_pnl.min(), (mu_g - 4*sigma_g)*start_capital))
    x_hi = float(max(hist_pnl_bhs.max() if len(hist_pnl_bhs) > 0 else 0, hist_pnl_boot.max(), mc_pnl.max(), (mu_g + 4*sigma_g)*start_capital))
    x_grid_C = np.linspace(x_lo, x_hi, 500)

    # KDEs für Hist (BHS), Bootstrapping und MC
    kde_hist_bhs = stats.gaussian_kde(hist_pnl_bhs) if len(hist_pnl_bhs) > 1 else None
    kde_hist_boot = stats.gaussian_kde(hist_pnl_boot)
    kde_mc = stats.gaussian_kde(mc_pnl)
    
    density_hist_bhs = kde_hist_bhs(x_grid_C) if kde_hist_bhs else np.zeros_like(x_grid_C)
    density_hist_boot = kde_hist_boot(x_grid_C)
    density_mc = kde_mc(x_grid_C)

    # Gauss-Dichte
    gauss_density = stats.norm.pdf(x_grid_C / start_capital, loc=mu_g, scale=sigma_g) / start_capital

    # VaR-Linien berechnen (mit der neuen PnL-Logik)
    var_bhs, _ = calculate_historical_risk(portfolio_returns_log, start_capital, alpha, days)
    var_boot, _ = calculate_bootstrap_risk(portfolio_returns_log, start_capital, alpha, days, simulations=simulations_boot)
    var_gauss, _ = calculate_gaussian_risk(portfolio_returns_discrete, start_capital, alpha, days)
    var_mc, _, _, _ = calculate_monte_carlo_risk(portfolio_returns_log, start_capital, alpha, days, simulations=simulations_boot)
    
    # Farbschema passend zur Präsentationslogik
    colors = {
        'BHS': 'rgb(133, 193, 233)',          
        'Boot': 'rgb(41, 128, 185)', 
        'Gauss': 'rgb(231, 76, 60)',                     
        'MC': 'rgb(35, 155, 86)'               
    }

    fig_pC = go.Figure()
    
    if len(hist_pnl_bhs) > 1:
        fig_pC.add_trace(go.Scatter(
            x=x_grid_C, y=density_hist_bhs, mode='lines',
            line=dict(color=colors['BHS'], width=2.2), name='Historisch (BHS)'
        ))
        
    fig_pC.add_trace(go.Scatter(
        x=x_grid_C, y=density_hist_boot, mode='lines',
        line=dict(color=colors['Boot'], width=2.2), name='Historisch (Bootstrapping)'
    ))
    fig_pC.add_trace(go.Scatter(
        x=x_grid_C, y=gauss_density, mode='lines',
        line=dict(color=colors['Gauss'], width=2.2), name='Gaußsch'
    ))
    fig_pC.add_trace(go.Scatter(
        x=x_grid_C, y=density_mc, mode='lines',
        line=dict(color=colors['MC'], width=2.2), name='Lognormal (MC)'
    ))
    
    # VaR Linien mit gestaffelten Beschriftungen
    if not np.isnan(var_bhs):
        fig_pC.add_vline(x=var_bhs, line=dict(color=colors['BHS'], width=1.5, dash='dash'))
        fig_pC.add_annotation(x=var_bhs, y=0.95, yref='paper', text=f"BHS: ${var_bhs:,.0f}", 
                              showarrow=False, font=dict(color=colors['BHS'], size=11), xanchor='right')

    fig_pC.add_vline(x=var_boot, line=dict(color=colors['Boot'], width=1.5, dash='dash'))
    fig_pC.add_annotation(x=var_boot, y=0.88, yref='paper', text=f"Bootstrapping: ${var_boot:,.0f}", 
                          showarrow=False, font=dict(color=colors['Boot'], size=11), xanchor='left')

    fig_pC.add_vline(x=var_gauss, line=dict(color=colors['Gauss'], width=1.5, dash='dash'))
    fig_pC.add_annotation(x=var_gauss, y=0.81, yref='paper', text=f"Gaußsch: ${var_gauss:,.0f}", 
                          showarrow=False, font=dict(color=colors['Gauss'], size=11), xanchor='left')

    fig_pC.add_vline(x=var_mc, line=dict(color=colors['MC'], width=1.5, dash='dash'))
    fig_pC.add_annotation(x=var_mc, y=0.74, yref='paper', text=f"MC: ${var_mc:,.0f}", 
                          showarrow=False, font=dict(color=colors['MC'], size=11), xanchor='right')

    conf_str = f"{(1-alpha)*100:.0f}"

    # Dynamischer Zoom auf die X-Achse
    mc_cutoff = np.percentile(mc_pnl, 95)
    hist_cutoff = hist_pnl_boot.max()
    zoom_max = max(mc_cutoff, hist_cutoff)
    zoom_max = max(zoom_max, start_capital * 0.5)

    fig_pC.update_layout(
        template='plotly_dark',
        title=f'Verteilungs- und Methodenvergleich ({horizon_label}) – {conf_str} % Konfidenz',
        xaxis_title='Gewinn/Verlust (USD)',
        yaxis_title='Wahrscheinlichkeitsdichte',
        height=520,
        hovermode='x unified',
        xaxis=dict(range=[x_lo, zoom_max])
    )
    
    return fig_pC

# ==========================================
# 5. UI HELPER FUNCTION
# ==========================================
def format_currency(value): #Formatierung für bessere Lesbarkeit der OUTPUTS
    if np.isnan(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:.0f}k"
    else:
        return f"${value:.0f}"

def render_risk_tab(days, tab_title):
    st.header(tab_title)
    
    # ==========================================
    # BEREICH A (Volle Breite)
    # ==========================================
    st.subheader("Auswahl VaR-Level - A")
    lvl_a_name = st.selectbox("VaR-Level A", list(var_levels_ui.keys()), key=f"sel_a_{days}")
    alpha_a = var_levels_ui[lvl_a_name]
    
    bhs_var_a, bhs_es_a = calculate_historical_risk(port_ret_log, start_capital, alpha_a, days)
    boot_var_a, boot_es_a = calculate_bootstrap_risk(port_ret_log, start_capital, alpha_a, days, simulations=10000)
    g_var_a, g_es_a = calculate_gaussian_risk(port_ret_discrete, start_capital, alpha_a, days)
    mc_var_a, mc_es_a, _, paths_a = calculate_monte_carlo_risk(port_ret_log, start_capital, alpha_a, days, simulations=2000, black_swan=False)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BHS VaR", format_currency(bhs_var_a))
    c1.metric("BHS ES", format_currency(bhs_es_a))
    c2.metric("Boot VaR", format_currency(boot_var_a))
    c2.metric("Boot ES", format_currency(boot_es_a))
    c3.metric("Gaußsch VaR", format_currency(g_var_a))
    c3.metric("Gaußsch ES", format_currency(g_es_a))
    c4.metric("MC VaR", format_currency(mc_var_a))
    c4.metric("MC ES", format_currency(mc_es_a))

    st.write("---") # Visuelle Trennung

    # ==========================================
    # BEREICH B (Volle Breite)
    # ==========================================
    st.subheader("Auswahl VaR-Level - B")
    lvl_b_name = st.selectbox("VaR-Level B", list(var_levels_ui.keys()), key=f"sel_b_{days}", index=1)
    alpha_b = var_levels_ui[lvl_b_name]
    
    bhs_var_b, bhs_es_b = calculate_historical_risk(port_ret_log, start_capital, alpha_b, days)
    boot_var_b, boot_es_b = calculate_bootstrap_risk(port_ret_log, start_capital, alpha_b, days, simulations=10000)
    g_var_b, g_es_b = calculate_gaussian_risk(port_ret_discrete, start_capital, alpha_b, days)
    mc_var_b, mc_es_b, _, paths_b = calculate_monte_carlo_risk(port_ret_log, start_capital, alpha_b, days, simulations=2000, black_swan=False)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BHS VaR", format_currency(bhs_var_b))
    c1.metric("BHS ES", format_currency(bhs_es_b))
    c2.metric("Boot VaR", format_currency(boot_var_b))
    c2.metric("Boot ES", format_currency(boot_es_b))
    c3.metric("Gaußsch VaR", format_currency(g_var_b))
    c3.metric("Gaußsch ES", format_currency(g_es_b))
    c4.metric("MC VaR", format_currency(mc_var_b))
    c4.metric("MC ES", format_currency(mc_es_b))

    st.write("---")
    
    # ==========================================
    # CHARTS (Nebeneinander ist hier okay, da Plots skalieren)
    # ==========================================
    col_chart_a, col_chart_b = st.columns(2)
    with col_chart_a:
        st.plotly_chart(plot_monte_carlo_fan_chart(paths_a, start_capital, alpha_a, f"MC Pfade & VaR A ({lvl_a_name})"), use_container_width=True)
    with col_chart_b:
        st.plotly_chart(plot_monte_carlo_fan_chart(paths_b, start_capital, alpha_b, f"MC Pfade & VaR B ({lvl_b_name})"), use_container_width=True)

# ==========================================
# 6. STREAMLIT APP LAYOUT
# ==========================================

# Sidebar
st.sidebar.title("Magnificent 7: Risiko - Dashboard")
start_capital = st.sidebar.number_input("Startkapital ($)", value=100_000, step=10_000)

# Tabs
tab_uebersicht, tab_1y, tab_5y, tab_10y, tab_black_swan, tab_methoden = st.tabs([
    "Übersicht", "1-Jahres-Risiko", "5-Jahres-Risiko", "10-Jahres-Risiko", "Black-Swan-Sim","Methodenvergleich"
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
   
    mkt_returns = bench_world.pct_change().dropna()
    sp500_returns = bench_sp500.pct_change().dropna()
    nasdaq_returns = bench_nasdaq.pct_change().dropna()

    
    fig_hist = plot_historical_performance(
     portfolio_returns = port_ret_discrete, 
     market_returns = mkt_returns, 
     sp500_returns = sp500_returns,
     nasdaq_returns = nasdaq_returns,
     start_capital = start_capital,
     title = "Performance MAG7 vs. Markt vs. SP500 vs. NASDAQ (2012-2026)"
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


# ----------------- REITER 5: METHODENVERGLEICH -----------------
with tab_methoden:
    st.header("Methodenvergleich: Historisch vs. Gaußsch vs. Lognormal")
    st.info("Dieser Tab beleuchtet die Unterschiede in den Berechnungsmodellen. Während die historische und Lognormal-Methode 'Fat Tails' (extreme Ränder) besser abbilden können, unterschätzt die Gauß-Verteilung diese häufig.")
    
    st.write("---")
    
    # ==========================
    # TEIL 1: BALKENDIAGRAMM (PLOT 1)
    # ==========================
    st.subheader("Übersicht über alle Anlagehorizonte")
    col_bar_1, col_bar_2 = st.columns(2)
    
    with col_bar_1:
        lvl_1_name = st.selectbox("Auswahl VaR-Level A", list(var_levels_ui.keys()), index=1, key="bar_lvl_1")
    with col_bar_2:
        lvl_2_name = st.selectbox("Auswahl VaR-Level B", list(var_levels_ui.keys()), index=0, key="bar_lvl_2")
        
    # Daten generieren. Wir passen die Namen an, damit sie zur Logik von get_comparison_data passen (95 % statt 5 %)
    conf_1_label = f"{int((1 - var_levels_ui[lvl_1_name]) * 100)} %"
    conf_2_label = f"{int((1 - var_levels_ui[lvl_2_name]) * 100)} %"
    
    df_comp = get_comparison_data(port_ret_log, port_ret_discrete, start_capital)
    
    # Plot 1 zeichnen
    fig_bar = plot_var_bar_comparison(df_comp, horizons, levels_to_show=[conf_1_label, conf_2_label])
    st.plotly_chart(fig_bar, use_container_width=True)

    st.warning(
        "**Analytischer Hinweis: Die Falle des Recency Bias im historischen Modell**\n\n"
        "Fällt dir auf, dass der **Historische VaR (Blau)** bei langen Horizonten oft viel geringere Risiken "
        "(oder sogar garantierte Gewinne) anzeigt als die anderen Modelle? Das ist ein klassischer **Recency Bias** "
        "(Verzerrung durch die jüngste Vergangenheit). \n\n"
        "Unser Datensatz (2012–2026) deckt fast ausschließlich den **beispiellosen Bullenmarkt der Tech-Branche** ab. "
        "Das historische Modell 'kennt' für die Magnificent 7 also faktisch keine echten, jahrelangen Bärenmärkte (wie z. B. nach 2000). "
        "Es nimmt blind an, dass die nächsten 10 Jahre genauso steigen wie die letzten. "
    )

    st.write("---")

    # ==========================
    # TEIL 2: DICHTEVERTEILUNG (PLOT C)
    # ==========================
    st.subheader("Dichteverteilung & VaR-Schwellen")
    
    col_dens_1, col_dens_2 = st.columns(2)
    with col_dens_1:
        # Dropdown für die Jahre (1, 5, 10, 20)
        selected_horizon_label = st.selectbox("Anlagehorizont wählen", list(horizons.keys()), key="dens_horizon")
        days_density = horizons[selected_horizon_label]
    
    with col_dens_2:
        # Dropdown für das VaR Level
        selected_var_ui = st.selectbox("VaR-Level wählen", list(var_levels_ui.keys()), index=1, key="dens_var")
        alpha_density = var_levels_ui[selected_var_ui]

    # Plot C zeichnen (mit Spinner, da MC + KDE leicht rechenintensiv sind)
    with st.spinner("Berechne Verteilungsdichten..."):
        fig_density = plot_density_comparison(
            port_ret_log, 
            port_ret_discrete, 
            start_capital, 
            days_density, 
            selected_horizon_label, 
            alpha_density
        )
    st.plotly_chart(fig_density, use_container_width=True)

