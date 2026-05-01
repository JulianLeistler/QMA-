import streamlit as st
import pandas as pd
import numpy as np
# ... (Hier kommen deine Plotly-Funktionen und Risiko-Funktionen rein) ...

# 1. Page Config (Macht das Dashboard breit und gibt ihm einen Titel)
st.set_page_config(page_title="MAG7 Risiko-Dashboard", layout="wide")

# 2. CACHING: Wir berechnen die aufwendigen Pfade nur EINMAL
@st.cache_data
def get_cached_monte_carlo_paths(capital, days, simulations=10000):
    """Holt die Lognormal-Pfade für das Normal-Szenario"""
    # Da var_level hier egal ist (wir wollen nur die Pfade), geben wir 0.05 als Dummy mit
    _, _, _, pf_paths = calculate_monte_carlo_risk_blackswan(
        portfolio_returns_log, capital, 0.05, days, simulations, black_swan=False
    )
    return pf_paths

@st.cache_data
def get_cached_black_swan_paths(capital, days, simulations=10000):
    """Holt die Lognormal-Pfade inkl. DotCom-Crash"""
    _, _, _, pf_paths = calculate_monte_carlo_risk_blackswan(
        portfolio_returns_log, capital, 0.05, days, simulations, 
        black_swan=True, crash_data=dotcom_returns
    )
    return pf_paths

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Parameter")
    
    # 1. Startkapital (Gibt den Wert '100000' als Integer zurück)
    start_capital = st.number_input(
        "Startkapital ($)", 
        min_value=10000, max_value=10000000, value=100000, step=10000
    )
    
    st.markdown("---")
    
    # 2. Für den Black Swan Reiter (Auswahl der Jahre)
    st.subheader("Black-Swan-Simulation")
    st.write("Wähle den Anlagehorizont für den Stresstest:")
    bs_horizon_years = st.selectbox("Anzahl Jahre", [1, 5, 10, 20])
    
    # Wir rechnen die Jahre für den Code direkt in Tage um (252 Handelstage)
    bs_horizon_days = bs_horizon_years * 252

# ---------------------------------------------------------
# HAUPTSEITE (Titel & Tabs)
# ---------------------------------------------------------
st.title("Magnificent 7: Risiko - Dashboard")

# Wir legen die 5 Reiter an
tab_uebersicht, tab_1j, tab_5j, tab_10j, tab_blackswan = st.tabs([
    "Übersicht", "1-Jahres-Risiko", "5-Jahres-Risiko", "10-Jahres-Risiko", "Black-Swan-Sim"
])

# ==========================================
# REITER 0: ÜBERSICHT
# ==========================================
with tab_uebersicht:
    st.header("Performance KPIs")
    
    # Hier rufst du deine Funktion calculate_performance_kpis auf
    # (Wir nehmen an, du hast das kpi_df vorliegen)
    
    # 4 Spalten für die KPIs anlegen
    col1, col2, col3, col4 = st.columns(4)
    
    # Die Metriken einfügen (mit den Tooltips/Helps, die du skizziert hast)
    # Beispielwerte, hier kommen später deine echten Variablen rein
    col1.metric("Beta", "1.15", help="Sensitivität gegenüber dem Markt.")
    col2.metric("Sharpe Ratio", "1.82", help="Überrendite pro Einheit Gesamtrisiko.")
    col3.metric("Roy's Safety First", "1.05", help="Risiko, den Markt zu underperformen.")
    col4.metric("Treynor Ratio", "0.2450", help="Überrendite pro Einheit Marktrisiko.")
    
    st.markdown("---")
    
    # Der Historische Chart (mit Spalte rechts, wie besprochen)
    col_chart, col_text = st.columns([0.7, 0.3])
    with col_chart:
        # Hier kommt dein historischer Chart rein
        # st.plotly_chart(mein_historischer_chart, use_container_width=True)
        st.info("Hier erscheint der Chart: MAG7 vs. Markt")
        
    with col_text:
        # Hier könnte dein DataFrame oder ein kleiner Erklärtext stehen
        st.write("Wachstums-Tabelle")
        
    st.markdown("---")
    st.subheader("Vollständige Risikomatrix")
    # st.dataframe(core_results)

# ==========================================
# REITER 1: 1-Jahres-Risiko (Beispielhaft für 1, 5 und 10)
# ==========================================
with tab_1j:
    # 2 Spalten für den Vergleich von VaR A und VaR B
    col_varA, col_varB = st.columns(2)
    
    with col_varA:
        st.subheader("Auswahl VaR-Level - A")
        var_a_selection = st.selectbox("Level A", ["5 %", "1 %", "10 %"], key="var_a_1j")
        # Hier baust du später deine kleine Tabelle ein
        st.write("Tabelle für Historisch, Gaußsch, Lognormal")
        
        # Hier kommt dein Plotly Fan-Chart hin
        st.info("Trichter-Chart A")
        
    with col_varB:
        st.subheader("Auswahl VaR-Level - B")
        var_b_selection = st.selectbox("Level B", ["1 %", "5 %", "10 %"], key="var_b_1j")
        st.write("Tabelle für Historisch, Gaußsch, Lognormal")
        
        st.info("Trichter-Chart B")

# (Die Reiter 5j und 10j sind im Prinzip Kopien von tab_1j)

# ==========================================
# REITER: Black-Swan-Simulation
# ==========================================
with tab_blackswan:
    st.markdown("""
    **Das Black-Swan-Event** simuliert den VaR & ES auf Basis eines zufällig eintretenden Crashes. 
    Dieses Event wurde mit den Renditen & Volatilitäten der DotCom-Blase simuliert.
    """)
    
    col_normal, col_swan = st.columns(2)
    
    with col_normal:
        st.subheader("Normal (Ohne Black Swan)")
        # Hier: Tabelle VaR & Chart
        st.info("Trichter-Chart 'Normal'")
        
    with col_swan:
        st.subheader("Black Swan (DotCom Crash)")
        # Hier: Tabelle VaR & Chart
        st.info("Trichter-Chart 'Black Swan'")

