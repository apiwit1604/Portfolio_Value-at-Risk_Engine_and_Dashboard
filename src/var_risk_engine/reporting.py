"""
Display helpers, kept separate from calculation logic (separation of
concerns) so the logic can be tested/modified without touching display code,
and vice versa.
"""

import matplotlib.pyplot as plt
import pandas as pd


def show_portfolio(name: str, portfolio: dict, tickers: list[str]) -> None:
    print("=" * 30)
    print(f" {name} ".center(30, " "))
    print("=" * 30)
    weights_df = pd.DataFrame(
        {"Weight (%)": portfolio["portfolio_weight"] * 100}, index=tickers
    ).round(2)
    print(weights_df)
    print(" Backtest (%) ".center(30, "="))
    print(f"Return        : {portfolio['portfolio_return'] * 100:.2f}")
    print(f"Risk          : {portfolio['portfolio_risk'] * 100:.2f}")
    print(f"Value at Risk : {portfolio['portfolio_var'] * 100:.2f}\n")


def show_var_comparison(title: str, values: dict[str, float]) -> None:
    width = 35
    print("=" * width)
    print(f" {title} ".center(width, " "))
    print("=" * width)
    for label, value in values.items():
        print(f"Value at Risk ({label:<8}): {value * 100:.2f}")
    print()


def plot_rolling_var(result_var: pd.DataFrame) -> None:
    """Plot actual portfolio return against each model's rolling VaR."""
    plt.figure(figsize=(10, 4))

    plt.plot(result_var.index, result_var["Actual Return"], label="Actual Return", linestyle="-", linewidth=2.5)
    plt.plot(result_var.index, result_var["Parametric VaR"], label="Parametric VaR", linestyle="--", linewidth=2.5)
    plt.plot(result_var.index, result_var["Historical VaR"], label="Historical VaR", linestyle="--", linewidth=2.5)
    plt.plot(result_var.index, result_var["Monte VaR"], label="Monte VaR", linestyle="--", linewidth=2.5)

    plt.title("Model VaR", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("return", fontsize=12)
    plt.margins(x=0)

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()
