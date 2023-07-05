"""
Copyright (c) 2023 Frederik Warburg and Marco Miani, Technical University of Denmark.
"""
from .hessian.contrastive import ContrastiveHessianCalculator
from .hessian.mse import MSEHessianCalculator
from .hessian.bce import BCEHessianCalculator
from .hessian.cross_entropy import CEHessianCalculator

from .optimization.prior_precision import (
    optimize_prior_precision,
    log_det_ratio,
    scatter,
    log_marginal_likelihood
)

from .laplace.diag import DiagLaplace
from .laplace.kron import BlockLaplace

__all__ = [
    "__version__",
    "ContrastiveHessianCalculator",
    "MSEHessianCalculator",
    "BCEHessianCalculator",
    "CEHessianCalculator",
    "optimize_prior_precision",
    "log_det_ratio",
    "scatter",
    "log_marginal_likelihood",
    "DiagLaplace",
    "BlockLaplace",
]