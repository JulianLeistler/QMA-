"""Plotting helpers for charts used in notebook and Streamlit.

This module should only format already computed data.
It should not contain primary model logic or hidden calculations.

The axis labels and plot titles defined here are the single source of truth
for the project's display standard (see MEMORY.md, Block "Display-Standard
fuer Plots"). UI code should never override these strings inline.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# --- Display standard --------------------------------------------------------

PLOT_TITLES: dict[str, str] = {
    "pnl_historical_1y": "Verteilung des Gewinns/Verlusts – Historische Methode – 1 Handelsjahr",
    "pnl_method_comparison_1y": "Verteilung des Gewinns/Verlusts – Methodenvergleich – 1 Handelsjahr",
    "endwert_quantiles_5y": "Quantile moeglicher Endwerte – Lognormal Monte Carlo – 5 Jahre",
    "correlation_matrix": "Korrelationsmatrix der Mag7-Tagesrenditen – 2012–2026",
    "var_estimation_uncertainty_1y": "Schaetzunsicherheit des VaR – 95 %-Konfidenz – 1 Handelsjahr",
    "simulation_paths_5y": "5.000 simulierte Wertentwicklungspfade – Lognormal Monte Carlo – 5 Jahre",
    "max_drawdown_5y": "Verteilung des maximalen Drawdowns – Lognormal Monte Carlo – 5 Jahre",
}

AXIS_LABELS: dict[str, str] = {
    "pnl_usd": "Gewinn/Verlust (USD)",
    "years": "Jahre",
    "var_usd": "VaR (USD)",
    "max_drawdown": "Maximaler Drawdown (%)",
    "density_per_usd_e5": "Dichte (×10⁻⁵ je USD)",
    "density_per_pct": "Dichte (je %-Punkt)",
    "endwert_usd": "Endwert (USD)",
    "portfolio_value_usd": "Portfoliowert (USD)",
}

PLOTLY_TEMPLATE = "plotly_white"
DEFAULT_MARGIN = dict(l=60, r=20, t=60, b=50)

# Design palette — portiert aus dem Matplotlib-Sandbox-Design in
# `leon_arbeitsstand/viz_prototypes.ipynb`. Wir bleiben fachlich beim
# Plotly-Stack; die Farbentscheidungen kommen aber aus dem ueberlegten
# Designsystem dort, damit Notebook und spaetere Streamlit-App einen
# konsistenten Look haben.
COLORS = {
    "primary": "#1e3a8a",      # tiefes Indigo, Hauptlinie / Median
    "accent": "#f43f5e",       # warmes Coral, Verlust / VaR / Tail
    "band_inner": "#3b82f6",   # helles Indigo, 25-75-Band
    "band_outer": "#dbeafe",   # sehr helles Indigo, 5-95-Band
    "neutral": "#64748b",      # Slate, Achsen / Tick-Labels
    "grid": "#e2e8f0",         # sehr helles Slate, Gridlines
    "method_hist": "#1e3a8a",  # Indigo
    "method_gauss": "#64748b", # Slate, "Referenz"
    "method_mc": "#a855f7",    # Violett, Akzent
}

# Plotly-Tickformat fuer USD-Werte. "$,.0f" zeigt z.B. -$25,000.
USD_TICKFORMAT = "$,.0f"


def apply_standard_layout(
    fig: go.Figure,
    title_key: str,
    x_label_key: str,
    y_label_key: str,
) -> go.Figure:
    """Set title, axis labels, template and margins from the registry.

    Raises KeyError on unknown keys so typos surface immediately rather than
    producing silently mislabelled plots.
    """
    fig.update_layout(
        title=PLOT_TITLES[title_key],
        xaxis_title=AXIS_LABELS[x_label_key],
        yaxis_title=AXIS_LABELS[y_label_key],
        template=PLOTLY_TEMPLATE,
        margin=DEFAULT_MARGIN,
    )
    return fig


# --- Plot 1: P/L distribution, historical, 1y -------------------------------

def plot_pnl_distribution_historical(
    pnl_usd: Sequence[float],
    var_usd: float | None = None,
    nbins: int = 60,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=list(pnl_usd),
            nbinsx=nbins,
            histnorm="probability density",
            name="Historisch",
            marker_color=COLORS["primary"],
            opacity=0.85,
        )
    )
    if var_usd is not None:
        fig.add_vline(
            x=var_usd,
            line_dash="dash",
            line_color=COLORS["accent"],
            annotation_text=f"VaR = {var_usd:,.0f} USD",
            annotation_font_color=COLORS["accent"],
        )
    fig = apply_standard_layout(fig, "pnl_historical_1y", "pnl_usd", "density_per_usd_e5")
    fig.update_xaxes(tickformat=USD_TICKFORMAT)
    return fig


# --- Plot 2: P/L distribution, method comparison, 1y ------------------------

_METHOD_COLOR_MAP = {
    "Historisch": COLORS["method_hist"],
    "Normal": COLORS["method_gauss"],
    "Lognormal MC": COLORS["method_mc"],
}


def plot_pnl_method_comparison(
    pnl_by_method: dict[str, Sequence[float]],
    nbins: int = 60,
) -> go.Figure:
    fig = go.Figure()
    for method_name, pnl in pnl_by_method.items():
        fig.add_trace(
            go.Histogram(
                x=list(pnl),
                nbinsx=nbins,
                histnorm="probability density",
                name=method_name,
                opacity=0.55,
                marker_color=_METHOD_COLOR_MAP.get(method_name),
            )
        )
    fig.update_layout(barmode="overlay")
    fig = apply_standard_layout(fig, "pnl_method_comparison_1y", "pnl_usd", "density_per_usd_e5")
    fig.update_xaxes(tickformat=USD_TICKFORMAT)
    return fig


# --- Plot 3: Endwert quantiles over horizon, MC, 5y -------------------------

def plot_endwert_quantiles(
    years: Sequence[float],
    quantiles: pd.DataFrame,
) -> go.Figure:
    """Quantile bands of the simulated end-value path.

    `quantiles` is expected as a DataFrame indexed by year with columns
    such as 'p05', 'p25', 'p50', 'p75', 'p95'.
    """
    fig = go.Figure()
    if {"p05", "p95"}.issubset(quantiles.columns):
        fig.add_trace(go.Scatter(
            x=years, y=quantiles["p95"], mode="lines", name="95 %-Quantil",
            line=dict(color=COLORS["band_outer"], width=1),
        ))
        fig.add_trace(go.Scatter(
            x=years, y=quantiles["p05"], mode="lines", name="5 %-Quantil",
            fill="tonexty", fillcolor="rgba(219, 234, 254, 0.6)",
            line=dict(color=COLORS["band_outer"], width=1),
        ))
    if {"p25", "p75"}.issubset(quantiles.columns):
        fig.add_trace(go.Scatter(
            x=years, y=quantiles["p75"], mode="lines", name="75 %-Quantil",
            line=dict(color=COLORS["band_inner"], width=1.2),
        ))
        fig.add_trace(go.Scatter(
            x=years, y=quantiles["p25"], mode="lines", name="25 %-Quantil",
            fill="tonexty", fillcolor="rgba(59, 130, 246, 0.25)",
            line=dict(color=COLORS["band_inner"], width=1.2),
        ))
    if "p50" in quantiles.columns:
        fig.add_trace(go.Scatter(
            x=years, y=quantiles["p50"], mode="lines", name="Median",
            line=dict(color=COLORS["primary"], width=2.5),
        ))
    fig = apply_standard_layout(fig, "endwert_quantiles_5y", "years", "endwert_usd")
    fig.update_yaxes(tickformat=USD_TICKFORMAT)
    return fig


# --- Plot 5: Correlation matrix ---------------------------------------------

def plot_correlation_matrix(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
        )
    )
    fig.update_layout(
        title=PLOT_TITLES["correlation_matrix"],
        template=PLOTLY_TEMPLATE,
        margin=DEFAULT_MARGIN,
    )
    return fig


# --- Plot 6: VaR estimation uncertainty, 1y ---------------------------------

def plot_var_estimation_uncertainty(
    var_estimates_usd: Sequence[float],
    nbins: int = 60,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=list(var_estimates_usd),
            nbinsx=nbins,
            histnorm="probability density",
            name="VaR-Schaetzungen",
            marker_color=COLORS["accent"],
            opacity=0.85,
        )
    )
    fig = apply_standard_layout(fig, "var_estimation_uncertainty_1y", "var_usd", "density_per_usd_e5")
    fig.update_xaxes(tickformat=USD_TICKFORMAT)
    return fig


# --- Plot 7: Simulated value paths, MC, 5y ----------------------------------

def plot_simulation_paths(
    years: Sequence[float],
    paths: np.ndarray,
    max_paths_to_draw: int = 5000,
) -> go.Figure:
    """Draw simulated value paths as a single concatenated trace.

    `paths` must be shape (n_paths, n_steps) and aligned with `years`.

    All paths are concatenated into one trace separated by NaNs so plotly
    breaks the line between paths. This keeps the notebook output orders of
    magnitude smaller than one-trace-per-path and is essential for the file
    to fit through git/GitHub.
    """
    years_arr = np.asarray(years, dtype=float)
    n_paths = paths.shape[0]
    n_to_draw = min(n_paths, max_paths_to_draw)
    selected = paths[:n_to_draw]

    n_steps = years_arr.size
    sep = np.full((n_to_draw, 1), np.nan)
    x_block = np.tile(np.concatenate([years_arr, [np.nan]]), n_to_draw)
    y_block = np.concatenate([selected, sep], axis=1).ravel()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_block,
            y=y_block,
            mode="lines",
            line=dict(width=0.5, color="rgba(59, 130, 246, 0.15)"),
            showlegend=False,
            hoverinfo="skip",
            name=f"{n_to_draw} Pfade",
        )
    )
    fig = apply_standard_layout(fig, "simulation_paths_5y", "years", "portfolio_value_usd")
    fig.update_yaxes(tickformat=USD_TICKFORMAT)
    return fig


# --- Plot 8: Max drawdown distribution, MC, 5y ------------------------------

def plot_max_drawdown_distribution(
    max_drawdowns: Sequence[float],
    nbins: int = 60,
) -> go.Figure:
    """Plot the distribution of the maximum drawdown.

    `max_drawdowns` must be passed as fractions (e.g. -0.25 for a 25% drawdown).
    The function converts to percent internally so the x-axis matches the
    "Dichte (je %-Punkt)" y-axis unit defined in the display standard.
    """
    drawdowns_pct = np.asarray(max_drawdowns, dtype=float) * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=drawdowns_pct,
            nbinsx=nbins,
            histnorm="probability density",
            name="Max Drawdown",
            marker_color=COLORS["accent"],
            opacity=0.85,
        )
    )
    return apply_standard_layout(fig, "max_drawdown_5y", "max_drawdown", "density_per_pct")
