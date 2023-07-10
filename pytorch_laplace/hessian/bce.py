from typing import Optional

import nnj
import torch
from backpack import extend
from torch import nn

from pytorch_laplace.hessian.base import HessianCalculator


class BCEHessianCalculator(HessianCalculator):
    "Binary Cross Entropy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lossfunc = extend(nn.BCEWithLogitsLoss())

    @torch.no_grad()
    def _compute_hessian_nnj(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor],
        model: nnj.Sequential,
    ) -> torch.Tensor:
        """
        Computes Generalized Gauss-Newton approximation (J^T H J) of the hessian of the network

        Args:
            x: input of the network
            target: output of the network (not used for bce loss)
            model: neural network module
        """

        val = model(x)

        bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
        H = bernoulli_p - bernoulli_p**2
        H = H.reshape(val.shape[0], -1)  # hessian in diagonal form

        # backpropagate through the network
        Jt_H_J = model.jTmjp(
            x,
            val,
            H,
            wrt="weight",
            from_diag=True,
            to_diag=self.hessian_shape == "diag",
            diag_backprop=self.approximation_accuracy == "exact",
        )
        # average along batch size
        Jt_H_J = torch.mean(Jt_H_J, dim=0)
        return Jt_H_J
