"""
risklib -- FinTech 545 risk library (functional tests 1.1 - 7.6).

Modules
-------
covariance   missing_cov, ew_covar, ew_correlation, ew_cov_var_corr, cov_to_corr
psd          near_psd, higham_nearest_psd, chol_psd
simulation   simulate_normal, simulate_pca
returns      return_calculate
fitting      fit_normal, fit_general_t, fit_regression_t, aicc,
             fit_nig_moments, fit_nig_mle, FittedModel
"""

from .covariance import (cov_to_corr, ew_cov_var_corr, ew_correlation, ew_covar,
                         ew_weights, missing_cov)
from .fitting import (FittedModel, aicc, fit_general_t, fit_nig_mle,
                      fit_nig_moments, fit_normal, fit_regression_t)
from .psd import chol_psd, higham_nearest_psd, near_psd
from .returns import return_calculate
from .simulation import simulate_normal, simulate_pca

__all__ = [
    "cov_to_corr", "ew_cov_var_corr", "ew_correlation", "ew_covar", "ew_weights",
    "missing_cov",
    "FittedModel", "aicc", "fit_general_t", "fit_nig_mle", "fit_nig_moments",
    "fit_normal", "fit_regression_t",
    "chol_psd", "higham_nearest_psd", "near_psd",
    "return_calculate",
    "simulate_normal", "simulate_pca",
]
