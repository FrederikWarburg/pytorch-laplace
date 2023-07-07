"""
Copyright (c) 2023 Frederik Warburg and Marco Miani, Technical University of Denmark.
"""
from .hessian.bce import BCEHessianCalculator
from .hessian.contrastive import ContrastiveHessianCalculator
from .hessian.cross_entropy import CEHessianCalculator
from .hessian.mse import MSEHessianCalculator
from .laplace.diag import DiagLaplace
from .laplace.kron import BlockLaplace
from .laplace.online_diag import OnlineDiagLaplace
from .laplace.online_kron import OnlineBlockLaplace
from .optimization.prior_precision import (
    log_det_ratio,
    log_marginal_likelihood,
    optimize_prior_precision,
    scatter,
)

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
    "OnlineDiagLaplace",
    "OnlineBlockLaplace",
]
