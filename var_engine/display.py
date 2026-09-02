# -*- coding: utf-8 -*-
"""Console / matplotlib display helpers (optional — used by examples/, not by app.py)."""

from __future__ import annotations


def display_portfolio_result(result, width=40):
    """Pretty-print weights, pricing, units, values, risk buckets, and all VaR types."""
    print("-" * width)
    print("weight_asset (per unit)")
    print("-" * width)
    print([f"{x:.4f}" for x in result["weight_asset"]])
    print(" ")

    print("-" * width)
    print("adjusted cashflow")
    print("-" * width)
    df = result["final_output"]
    print(df.to_string(formatters={"adj_cf": lambda x: f"{x:.2f}", "adj_weight": lambda x: f"{x:.6f}"}))
    print()

    print("-" * width)
    print("Indicator portfolio")
    print("-" * width)
    print(f"ESG Score        : {result['esg_score']:.2f}")
    print(f"Sharp Ratio      : {result['sharp']:.2f}")
    print(f"Portfolio Return : {result['portfolio_return'] * 100:.2f}%")
    print(f"Portfolio Risk   : {result['portfolio_risk'] * 100:.2f}%")
    print()

    print("-" * width)
    print("Value at Risk portfolio")
    print("-" * width)
    print(f"Parametric       : {result['portfolio_var_parametric']:.4f}")
    print(f"Historical       : {result['portfolio_var_historical']:.4f}")
    print(f"Monte Carlo      : {result['portfolio_var_mc']:.4f}")


def plot_portfolio_var_breaches(df, return_col="return_port", var_cols=None,
                                 figsize=(14, 7), title="Portfolio Returns vs. Value at Risk (VaR)"):
    """Plot realized portfolio returns against each VaR line, marking every breach with an 'x'."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if var_cols is None:
        var_cols = {
            "var_parametric": ("##FF6F00", "--", "Parametric VaR"),
            "var_historical": ("#CCFF00", "--", "Historical VaR"),
            "var_mc": ("#7fff00", "--", "Monte Carlo VaR"),
        }

    plot_data = df.copy()

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(plot_data.index, plot_data[return_col], label="Portfolio Return",
            color="black", linewidth=1.2, alpha=0.6)

    for col, (color, linestyle, label_name) in var_cols.items():
        if col not in plot_data.columns:
            continue

        var_series = plot_data[col]
        ax.plot(plot_data.index, var_series, label=label_name, linestyle=linestyle,
                color=color, alpha=0.8, linewidth=1.5)

        breaches = plot_data[plot_data[return_col] < var_series]
        if not breaches.empty:
            ax.scatter(breaches.index, breaches[return_col], color="red", marker="x", s=25,
                       zorder=5, label=f"Breach ({label_name})" if len(var_cols) == 1 else None)

    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Return / VaR", fontsize=11)
    ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)
    ax.margins(x=0)
    fig.tight_layout()
    return fig

def plot_portfolio_var_breaches_interactive(df, return_col="return_port", var_cols=None,
                                             title="Portfolio Returns vs. Value at Risk (VaR)"):
    """
    Plotly version of plot_portfolio_var_breaches() — used by the Streamlit
    dashboard (st.plotly_chart) instead of the static matplotlib figure so
    the user can hover any date to see the exact return/VaR values, zoom
    into a date range, and toggle series on/off by clicking the legend.

    Same breach logic as the matplotlib version (a breach = realized
    return below that day's VaR line for a given method); returns a
    plotly.graph_objects.Figure.
    """
    import plotly.graph_objects as go

    if var_cols is None:
        var_cols = {
            "var_parametric": ("#1f77b4", "Parametric VaR"),
            "var_historical": ("#2ca02c", "Historical VaR"),
            "var_mc": ("#ff7f0e", "Monte Carlo VaR"),
        }

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index, y=df[return_col], mode="lines", name="Portfolio Return",
        line=dict(color="red", width=1.5),
        hovertemplate="%{y:.2%}<extra>Portfolio Return</extra>",
    ))

    for col, (color, label) in var_cols.items():
        if col not in df.columns:
            continue

        var_series = df[col]
        fig.add_trace(go.Scatter(
            x=df.index, y=var_series, mode="lines", name=label,
            line=dict(color=color, dash="dash", width=1.5),
            hovertemplate="%{y:.2%}<extra>" + label + "</extra>",
        ))

        breaches = df[df[return_col] < var_series]
        if not breaches.empty:
            fig.add_trace(go.Scatter(
                x=breaches.index, y=breaches[return_col], mode="markers",
                name=f"Breach ({label})",
                marker=dict(color="red", symbol="x", size=9, line=dict(width=1)),
                hovertemplate="%{y:.2%}<extra>Breach — " + label + "</extra>",
            ))

    fig.add_hline(y=0, line=dict(color="gray", width=1))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Return / VaR",
        yaxis_tickformat=".2%",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=40),
        height=500,
    )
    return fig
