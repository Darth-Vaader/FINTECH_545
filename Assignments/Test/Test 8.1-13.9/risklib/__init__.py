"""
risklib -- FinTech 545 risk library (functional tests 8.1 - 13.9).

Includes earlier helpers (fitting, simulation, PSD) that these tests depend on.
Tests 1.1-7.6 live in the sibling folder with their own copy of the library.
"""

from .attribution import expost_factor
from .copula import (copula_aicc, copula_bic, fit_gaussian_copula, fit_t_copula,
                     simulate_gaussian_copula, simulate_t_copula,
                     tail_dependence_t)
from .covariance import (cov_to_corr, ew_cov_var_corr, ew_correlation, ew_covar,
                         ew_weights, missing_cov)
from .fitting import (FittedModel, aicc, fit_general_t, fit_nig_mle,
                      fit_nig_moments, fit_normal, fit_regression_t)
from .multivariate_t import fit_multivariate_t, kendall_correlation
from .optimizers import max_sr, risk_parity
from .options import (american_greeks_fd, bt_american, bt_american_div, gbsm)
from .psd import chol_psd, higham_nearest_psd, near_psd
from .returns import return_calculate
from .risk_stats import ES, VaR, agg_risk
from .simulation import simulate_normal, simulate_pca

__all__ = [
    "expost_factor",
    "copula_aicc", "copula_bic", "fit_gaussian_copula", "fit_t_copula",
    "simulate_gaussian_copula", "simulate_t_copula", "tail_dependence_t",
    "cov_to_corr", "ew_cov_var_corr", "ew_correlation", "ew_covar", "ew_weights",
    "missing_cov",
    "FittedModel", "aicc", "fit_general_t", "fit_nig_mle", "fit_nig_moments",
    "fit_normal", "fit_regression_t",
    "fit_multivariate_t", "kendall_correlation",
    "max_sr", "risk_parity",
    "american_greeks_fd", "bt_american", "bt_american_div", "gbsm",
    "chol_psd", "higham_nearest_psd", "near_psd",
    "return_calculate",
    "ES", "VaR", "agg_risk",
    "simulate_normal", "simulate_pca",
]
