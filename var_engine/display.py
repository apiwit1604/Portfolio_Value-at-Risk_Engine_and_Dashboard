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
            "var_parametric": ("#1f77b4", "--", "Parametric VaR"),
            "var_historical": ("#2ca02c", "--", "Historical VaR"),
            "var_mc": ("#ff7f0e", "--", "Monte Carlo VaR"),
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
