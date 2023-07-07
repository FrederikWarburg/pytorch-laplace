from typing import Optional

import nnj
import torch

from pytorch_laplace.hessian.base import HessianCalculator


class BCEHessianCalculator(HessianCalculator):
    "Binary Cross Entropy"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        model: nnj.Sequential,
    ) -> torch.Tensor:
        """
        Computes Binary Cross Entropy

        Args:
            x: input of the network
            target: output of the network
            model: neural network module
        """

        val = model(x)
        assert val.shape == target.shape

        bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
        cross_entropy = -(target * torch.log(bernoulli_p) + (1 - target) * torch.log(1 - bernoulli_p))

        # average along batch size
        cross_entropy = torch.mean(cross_entropy, dim=0)
        # sum along other dimensions
        cross_entropy = torch.sum(cross_entropy)
        return cross_entropy

    @torch.no_grad()
    def compute_gradient(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        model: nnj.Sequential,
    ) -> torch.Tensor:
        """
        Computes gradient of the network

        Args:
            x: input of the network
            target: output of the network
            model: neural network module
        """

        val = model(x)
        assert val.shape == target.shape

        bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
        gradient = bernoulli_p - target

        # backpropagate through the network
        gradient = gradient.reshape(val.shape[0], -1)
        gradient = model._vjp(x, val, gradient, wrt="weight")

        # average along batch size
        gradient = torch.mean(gradient, dim=0)
        return gradient

    @torch.no_grad()
    def compute_hessian(
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
