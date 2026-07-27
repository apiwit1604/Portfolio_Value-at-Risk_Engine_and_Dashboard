"""
Shared fixtures for the test suite.

All tests use synthetic (seeded, reproducible) data rather than live network
calls to Yahoo Finance -- the test suite must run offline and deterministically.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tickers():
    return ["A", "B", "C"]


@pytest.fixture
def synthetic_returns(tickers):
    """~3 years of daily log returns for 3 synthetic assets with known,
    controlled mean/covariance structure.

    Asset A: low mean, low variance
    Asset B: mid mean, mid variance
    Asset C: high mean, high variance
    Correlations are deliberately non-zero so covariance-based calculations
    are actually exercised.
    """
    rng = np.random.default_rng(seed=7)
    n_days = 750

    mu_true = np.array([0.0002, 0.0005, 0.0009])
    cov_true = np.array(
        [
            [0.00010, 0.00003, 0.00002],
            [0.00003, 0.00025, 0.00005],
            [0.00002, 0.00005, 0.00060],
        ]
    )

    data = rng.multivariate_normal(mu_true, cov_true, size=n_days)
    dates = pd.bdate_range("2021-01-01", periods=n_days)
    return pd.DataFrame(data, index=dates, columns=tickers)


@pytest.fixture
def equal_weights(tickers):
    n = len(tickers)
    return np.array([1 / n] * n)
