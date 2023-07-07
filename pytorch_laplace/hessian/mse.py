from typing import Optional

import nnj
import torch

from pytorch_laplace.hessian.base import HessianCalculator


class MSEHessianCalculator(HessianCalculator):
    "Mean Square Error"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def compute_loss(self, x: torch.Tensor, target: torch.Tensor, model: nnj.Sequential) -> torch.Tensor:
        """
        Computes Mean Square Error

        Args:
            x: input of the network
            target: output of the network
            model: neural network module
        """

        val = model(x)
        assert val.shape == target.shape

        # compute Gaussian log-likelihood
        mse = 0.5 * (val - target) ** 2

        # average along batch size
        mse = torch.mean(mse, dim=0)

        # sum along other dimensions
        mse = torch.sum(mse)

        return mse

    @torch.no_grad()
    def compute_gradient(self, x: torch.Tensor, target: torch.Tensor, model: nnj.Sequential) -> torch.Tensor:
        """
        Computes gradient of the network

        Args:
            x: input of the network
            target: output of the network
            model: neural network module
        """

        val = model(x)
        assert val.shape == target.shape

        # compute gradient of the Gaussian log-likelihood
        gradient = val - target

        # backpropagate through the network
        gradient = gradient.reshape(val.shape[0], -1)
        gradient = model.vjp(x, val, gradient, wrt="weight")

        # average along batch size
        gradient = torch.mean(gradient, dim=0)
        return gradient

    @torch.no_grad()
    def compute_hessian(
        self, x: torch.Tensor, model: nnj.Sequential, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute generalized-gauss newton (GGN) approximation of the Hessian.
        Jacobian sandwich of the identity for each element in the batch
        H = identity matrix (None is interpreted as identity by jTmjp)

        Args:
            x: input of the network
            target: output of the network (not used for MSE)
            model: neural network module
        """
        val = model(x)

        # backpropagate through the network
        Jt_J = model.jTmjp(
            x,
            val,
            None,
            wrt="weight",
            to_diag=self.hessian_shape == "diag",
            diag_backprop=self.approximation_accuracy == "approx",
        )
        # average along batch size
        Jt_J = torch.mean(Jt_J, dim=0)
        return Jt_J
