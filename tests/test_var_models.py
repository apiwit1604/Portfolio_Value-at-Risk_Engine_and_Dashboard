"""
Tests for var_risk_engine.var_models.

These tests target the properties that are easy to silently break during
refactoring: sign convention, scaling behavior, and agreement between
independent implementations (parametric vs. Monte Carlo on Gaussian data).
"""

import numpy as np
import pytest
from scipy.stats import norm

from var_risk_engine.portfolio_stats import scale_cov, scale_mu
from var_risk_engine.var_models import historical_var, monte_carlo_var, performance_portfolio


class TestPerformancePortfolio:
    def test_var_is_negative_for_typical_confidence(self, synthetic_returns, equal_weights):
        """At 95% confidence with realistic mu/sigma, VaR should come out
        negative (a loss), not positive. This is the sign convention the
        rest of the codebase (optimization, backtest) depends on.
        """
        mu = scale_mu(synthetic_returns.mean(), horizon=5)
        sigma = scale_cov(synthetic_returns.cov(), horizon=5)

        result = performance_portfolio(equal_weights, mu, sigma, confidence=0.95)

        assert result["portfolio_var"] < 0

    def test_var_formula_matches_manual_calculation(self, synthetic_returns, equal_weights):
        mu = scale_mu(synthetic_returns.mean(), horizon=5)
        sigma = scale_cov(synthetic_returns.cov(), horizon=5)
        confidence = 0.95

        result = performance_portfolio(equal_weights, mu, sigma, confidence)

        expected_return = equal_weights @ mu
        expected_risk = np.sqrt(equal_weights @ sigma @ equal_weights)
        expected_z = norm.ppf(1 - confidence)
        expected_var = expected_return + expected_z * expected_risk

        assert result["portfolio_return"] == pytest.approx(expected_return)
        assert result["portfolio_risk"] == pytest.approx(expected_risk)
        assert result["portfolio_var"] == pytest.approx(expected_var)

    def test_higher_confidence_gives_more_negative_var(self, synthetic_returns, equal_weights):
        """VaR at 99% confidence should be worse (more negative) than at
        95%, since we're moving further into the tail.
        """
        mu = scale_mu(synthetic_returns.mean(), horizon=5)
        sigma = scale_cov(synthetic_returns.cov(), horizon=5)

        var_95 = performance_portfolio(equal_weights, mu, sigma, 0.95)["portfolio_var"]
        var_99 = performance_portfolio(equal_weights, mu, sigma, 0.99)["portfolio_var"]

        assert var_99 < var_95

    def test_concentrating_in_riskiest_asset_increases_risk(self, synthetic_returns):
        """Sanity check on the risk formula: putting all weight on the
        highest-variance asset (C, index 2) must give higher portfolio risk
        than equal weighting.
        """
        mu = scale_mu(synthetic_returns.mean(), horizon=1)
        sigma = scale_cov(synthetic_returns.cov(), horizon=1)

        w_equal = np.array([1 / 3, 1 / 3, 1 / 3])
        w_concentrated = np.array([0.0, 0.0, 1.0])

        risk_equal = performance_portfolio(w_equal, mu, sigma, 0.95)["portfolio_risk"]
        risk_concentrated = performance_portfolio(w_concentrated, mu, sigma, 0.95)["portfolio_risk"]

        assert risk_concentrated > risk_equal


class TestScaling:
    def test_mean_scales_linearly_with_horizon(self, synthetic_returns):
        daily_mean = synthetic_returns.mean()
        assert scale_mu(daily_mean, horizon=5).equals(daily_mean * 5)

    def test_covariance_scales_linearly_with_horizon(self, synthetic_returns):
        daily_cov = synthetic_returns.cov()
        scaled = scale_cov(daily_cov, horizon=5)
        assert np.allclose(scaled.to_numpy(), (daily_cov * 5).to_numpy())

    def test_longer_horizon_gives_larger_risk(self, synthetic_returns, equal_weights):
        """Risk (portfolio std) must grow with sqrt(horizon), so a 25-day
        horizon should give exactly 5x the risk of a 1-day horizon
        (since sqrt(25) = 5).
        """
        mu_1d = scale_mu(synthetic_returns.mean(), horizon=1)
        sigma_1d = scale_cov(synthetic_returns.cov(), horizon=1)
        mu_25d = scale_mu(synthetic_returns.mean(), horizon=25)
        sigma_25d = scale_cov(synthetic_returns.cov(), horizon=25)

        risk_1d = performance_portfolio(equal_weights, mu_1d, sigma_1d, 0.95)["portfolio_risk"]
        risk_25d = performance_portfolio(equal_weights, mu_25d, sigma_25d, 0.95)["portfolio_risk"]

        assert risk_25d == pytest.approx(risk_1d * 5, rel=1e-8)


class TestHistoricalVar:
    def test_matches_pandas_quantile_directly(self, synthetic_returns, equal_weights):
        alpha = 0.05
        result = historical_var(synthetic_returns, equal_weights, alpha)
        expected = (synthetic_returns @ equal_weights).quantile(alpha)
        assert result == pytest.approx(expected)

    def test_lower_alpha_gives_more_negative_var(self, synthetic_returns, equal_weights):
        """The 1% quantile of returns should be more negative than the 5%
        quantile -- basic monotonicity of the empirical quantile function.
        """
        var_1pct = historical_var(synthetic_returns, equal_weights, alpha=0.01)
        var_5pct = historical_var(synthetic_returns, equal_weights, alpha=0.05)
        assert var_1pct < var_5pct


class TestMonteCarloVar:
    def test_converges_to_parametric_var_on_gaussian_data(self, synthetic_returns, equal_weights):
        """Monte Carlo simulates returns from the SAME Gaussian assumptions
        the parametric model uses analytically. With enough simulations,
        the two should agree closely -- this catches sign errors, scaling
        errors, or an inverted quantile direction in either implementation.
        """
        np.random.seed(123)
        horizon = 5
        confidence = 0.95
        alpha = 1 - confidence

        mu = scale_mu(synthetic_returns.mean(), horizon)
        sigma = scale_cov(synthetic_returns.cov(), horizon)
        parametric = performance_portfolio(equal_weights, mu, sigma, confidence)["portfolio_var"]

        mu_daily = synthetic_returns.mean().to_numpy()
        sigma_daily = synthetic_returns.std().to_numpy()
        corr = synthetic_returns.corr().to_numpy()

        mc_var = monte_carlo_var(
            mu_daily, sigma_daily, corr, equal_weights, horizon, alpha, n_sims=200_000
        )

        # Monte Carlo has sampling noise; allow a reasonably wide absolute tolerance.
        assert mc_var == pytest.approx(parametric, abs=0.01)

    def test_is_reproducible_with_seeded_rng(self, synthetic_returns, equal_weights):
        mu_daily = synthetic_returns.mean().to_numpy()
        sigma_daily = synthetic_returns.std().to_numpy()
        corr = synthetic_returns.corr().to_numpy()

        np.random.seed(42)
        first = monte_carlo_var(mu_daily, sigma_daily, corr, equal_weights, 5, 0.05, n_sims=10_000)
        np.random.seed(42)
        second = monte_carlo_var(mu_daily, sigma_daily, corr, equal_weights, 5, 0.05, n_sims=10_000)

        assert first == second
