from abc import ABC, abstractmethod
from typing import Literal, Optional

import nnj
import torch
from backpack import backpack
from backpack.extensions import DiagGGNExact
from torch import nn


class HessianCalculator(ABC, nn.Module):
    """
    Base class for Hessian calculators
    """

    def __init__(
        self,
        hessian_shape: Literal["full", "block", "diag"] = "diag",
        approximation_accuracy: Literal["exact", "approx"] = "exact",
        backend: Literal["backpack", "nnj"] = "nnj",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        assert hessian_shape in ("full", "block", "diag")
        assert approximation_accuracy in ("exact", "approx")
        assert backend in ("backpack", "nnj")

        self.hessian_shape = hessian_shape
        self.approximation_accuracy = approximation_accuracy
        self.backend = backend

    def compute_hessian(
        self, x: torch.Tensor, model: nnj.Sequential, target: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Hessian of the MSE loss function
        using either the backpack or nnj backend

        Args:
            x: input of the network
            target: output of the network (not used for MSE)
            model: neural network module
        """
        if self.backend == "backpack":
            dggn = self._compute_hessian_backpack(x, model, target)
        elif self.backend == "nnj":
            dggn = self._compute_hessian_nnj(x, model, target)
        else:
            raise ValueError(f"Unknown backend {self.backend}")

        return dggn

    def _compute_hessian_backpack(
        self, x: torch.Tensor, model: nn.Sequential, target: torch.Tensor, *args, **kwargs
    ):
        """
        Compute generalized-gauss newton (GGN) approximation of the Hessian
        using the backpack backend.

        Args:
            x: input of the network
            target: output of the network (not used for MSE)
            model: neural network module
        """

        f = model(x)

        if not hasattr(self, "lossfunc"):
            raise ValueError("Loss function not set. Please set self.lossfunc")

        loss = self.lossfunc(f, target)
        with backpack(DiagGGNExact()):
            loss.backward()
        dggn = torch.cat([p.diag_ggn_exact.data.flatten() for p in model.parameters()])

        return dggn

    @abstractmethod
    @torch.no_grad()
    def _compute_hessian_nnj(
        self, x: torch.Tensor, target: Optional[torch.Tensor], model: nnj.Sequential, *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute generalized-gauss newton (GGN) approximation of the Hessian
        using the nnj backend.

        Args:
            x: input of the network
            target: output of the network (not used for MSE)
            model: neural network module
        """

        raise NotImplementedError
