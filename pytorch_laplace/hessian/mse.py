from typing import Literal, Optional

import nnj
import torch
from backpack import extend
from torch import nn

from pytorch_laplace.hessian.base import HessianCalculator


class MSEHessianCalculator(HessianCalculator):
    "Mean Square Error"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lossfunc = extend(nn.MSELoss())

    @torch.no_grad()
    def _compute_hessian_nnj(
        self, x: torch.Tensor, model: nnj.Sequential, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute generalized-gauss newton (GGN) approximation of the Hessian
        using the nnj backend.
        Jacobian sandwich of the identity for each element in the batch
        H = identity matrix (None is interpreted as identity by jTmjp)

        Args:
            x: input of the network
            target: output of the network (not used for MSE)
            model: neural network module
        """
        val = model(x)

        # backpropagate through the network
        dggn = model.jTmjp(
            x,
            val,
            None,  # none is interpreted as identity matrix
            wrt="weight",
            to_diag=self.hessian_shape == "diag",
            diag_backprop=self.approximation_accuracy == "approx",
        )
        # average along batch size
        dggn = torch.mean(dggn, dim=0)
        return dggn
