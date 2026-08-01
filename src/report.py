"""
Reporting Helpers
=================
Display helpers, kept separate from calculation logic (separation of
concerns) so the logic can be tested/modified without affecting display,
and vice versa.
"""

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
