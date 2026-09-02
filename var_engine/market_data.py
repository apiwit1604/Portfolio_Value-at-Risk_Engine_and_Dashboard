# -*- coding: utf-8 -*-
"""Market data: U.S. Treasury yield curve (FRED) download + interpolation."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from .cache import fetch_fred_series

# Maturity (years) -> FRED series ID. Add/remove tenors here for a
# finer/coarser curve; a cubic spline is fit through whatever tenors
# you give it (see interpolate_yield_curve).
DEFAULT_FRED_SERIES = {
    1 / 12: "DGS1MO",
    3 / 12: "DGS3MO",
    6 / 12: "DGS6MO",
    1.0: "DGS1",
    2.0: "DGS2",
    3.0: "DGS3",
    5.0: "DGS5",
    7.0: "DGS7",
    10.0: "DGS10",
    20.0: "DGS20",
    30.0: "DGS30",
}


def get_fred_yield_curve(start_date, end_date, fred_series: dict | None = None):
    """
    Download U.S. Treasury yields from FRED and align them into one
    date x maturity table of decimal yields.

    Parameters
    ----------
    start_date : str
        Start date, e.g. "2020-01-01".
    end_date : str or None
        End date, e.g. "2025-12-31". `None` means "up to today".
    fred_series : dict, optional
        Mapping from maturity in years to FRED series ID.
        Defaults to DEFAULT_FRED_SERIES.

    Returns
    -------
    data : pd.DataFrame
        Index = observation date, columns = maturity in years,
        values = decimal yields (e.g. 4.5% -> 0.045).
    x_known : np.ndarray
        The known maturities in years (== fred_series.keys()).
    """
    if fred_series is None:
        fred_series = DEFAULT_FRED_SERIES

    if end_date is None:
        # end_date=None interpolated literally into the URL as "None"
        # (&coed=None) breaks FRED's CSV endpoint. Default to today.
        end_date = datetime.today().strftime("%Y-%m-%d")

    data = pd.DataFrame()
    for maturity_yr, series_id in fred_series.items():
        series = fetch_fred_series(series_id, start_date, end_date)
        data = data.join(series.rename(maturity_yr), how="outer")

    data = data.sort_index()
    data = data.ffill()   # carry the last known yield forward over holidays/gaps
    data = data.dropna()  # keep only dates where every tenor has a value
    data = data / 100     # FRED yields are quoted in percentage points

    x_known = np.array(list(fred_series.keys()))
    return data, x_known


def interpolate_yield_curve(row, x_known, x_target):
    """
    Fit a cubic spline through one date's known par yields and evaluate
    it at x_target (one or more maturities, in years).

    Always returns a pd.Series — callers that need a single scalar
    (e.g. get_zero_bond) must extract it explicitly with .iloc[0].

    Caveat: x_target values outside [min(x_known), max(x_known)]
    (currently ~1 month to 30 years with the default tenor set) are
    extrapolated by the spline and can behave unpredictably. Keep
    bond/option maturities inside that range.
    """
    y_known = row.values
    interpolator = CubicSpline(x_known, y_known)
    return pd.Series(interpolator(x_target))
