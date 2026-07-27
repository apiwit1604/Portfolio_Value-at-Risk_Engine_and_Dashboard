"""
Tests for var_risk_engine.optimization.

The critical regression to guard against here is the sign-convention bug
this project already had once: find_min_loss_portfolio used to be named
find_min_var_portfolio, which read as "minimize VaR" (i.e. drive toward the
worst possible loss) when the function actually maximizes VaR (minimizes
tail loss). These tests pin down the actual optimization DIRECTION, not
just that the optimizer runs without error.
"""

import numpy as np
import pytest

from var_risk_engine.optimization import find_min_loss_portfolio, find_min_risk_portfolio
from var_risk_engine.portfolio_stats import scale_cov, scale_mu
from var_risk_engine.var_models import performance_portfolio


@pytest.fixture
def mu_sigma(synthetic_returns):
    mu = scale_mu(synthetic_returns.mean(), horizon=5)
    sigma = scale_cov(synthetic_returns.cov(), horizon=5)
    return mu, sigma


class TestFindMinRiskPortfolio:
    def test_weights_sum_to_one(self, mu_sigma, equal_weights):
        mu, sigma = mu_sigma
        result = find_min_risk_portfolio(mu, sigma, equal_weights, 0, 1, 0.95)
        assert result["portfolio_weight"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_respect_bounds(self, mu_sigma, equal_weights):
        mu, sigma = mu_sigma
        result = find_min_risk_portfolio(mu, sigma, equal_weights, 0.1, 0.5, 0.95)
        weights = result["portfolio_weight"]
        assert np.all(weights >= 0.1 - 1e-6)
        assert np.all(weights <= 0.5 + 1e-6)

    def test_achieves_lower_or_equal_risk_than_equal_weight(self, mu_sigma, equal_weights):
        """The whole point of this optimizer is that it should never do
        WORSE than an arbitrary starting point on its own objective (risk).
        """
        mu, sigma = mu_sigma
        equal_weight_risk = performance_portfolio(equal_weights, mu, sigma, 0.95)["portfolio_risk"]
        result = find_min_risk_portfolio(mu, sigma, equal_weights, 0, 1, 0.95)

        assert result["portfolio_risk"] <= equal_weight_risk + 1e-8

    def test_favors_lowest_variance_asset(self, synthetic_returns):
        """With three assets of clearly different variances (see
        conftest.synthetic_returns: A is low-variance, C is high-variance),
        the minimum-variance portfolio should allocate more to A than to C.
        """
        mu = scale_mu(synthetic_returns.mean(), horizon=1)
        sigma = scale_cov(synthetic_returns.cov(), horizon=1)
        w0 = np.array([1 / 3, 1 / 3, 1 / 3])

        result = find_min_risk_portfolio(mu, sigma, w0, 0, 1, 0.95)
        weight_a, _weight_b, weight_c = result["portfolio_weight"]

        assert weight_a > weight_c


class TestFindMinLossPortfolio:
    def test_weights_sum_to_one(self, mu_sigma, equal_weights):
        mu, sigma = mu_sigma
        result = find_min_loss_portfolio(mu, sigma, equal_weights, 0, 1, 0.95)
        assert result["portfolio_weight"].sum() == pytest.approx(1.0, abs=1e-6)

    def test_achieves_var_at_least_as_good_as_equal_weight(self, mu_sigma, equal_weights):
        """Core direction check: find_min_loss_portfolio must produce a VaR
        that is >= (less negative / less severe loss than) the equal-weight
        VaR. If this ever flips (VaR gets MORE negative), the optimizer is
        pointed the wrong way -- exactly the bug the old function name
        invited.
        """
        mu, sigma = mu_sigma
        equal_weight_var = performance_portfolio(equal_weights, mu, sigma, 0.95)["portfolio_var"]
        result = find_min_loss_portfolio(mu, sigma, equal_weights, 0, 1, 0.95)

        assert result["portfolio_var"] >= equal_weight_var - 1e-8

    def test_min_loss_var_is_not_worse_than_min_risk_var(self, mu_sigma, equal_weights):
        """find_min_loss_portfolio directly optimizes VaR, while
        find_min_risk_portfolio optimizes risk only (ignoring return). Since
        min-loss explicitly targets the VaR objective, its VaR should be at
        least as good as whatever min-risk happens to achieve.
        """
        mu, sigma = mu_sigma
        min_risk_result = find_min_risk_portfolio(mu, sigma, equal_weights, 0, 1, 0.95)
        min_loss_result = find_min_loss_portfolio(mu, sigma, equal_weights, 0, 1, 0.95)

        assert min_loss_result["portfolio_var"] >= min_risk_result["portfolio_var"] - 1e-8

    def test_weights_respect_bounds(self, mu_sigma, equal_weights):
        mu, sigma = mu_sigma
        result = find_min_loss_portfolio(mu, sigma, equal_weights, 0.05, 0.6, 0.95)
        weights = result["portfolio_weight"]
        assert np.all(weights >= 0.05 - 1e-6)
        assert np.all(weights <= 0.6 + 1e-6)
