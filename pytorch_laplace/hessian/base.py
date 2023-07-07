from abc import ABC, abstractmethod
from typing import Literal, Optional

import nnj
import torch
from torch import nn


class HessianCalculator(ABC, nn.Module):
    def __init__(
        self,
        hessian_shape: Literal["full", "block", "diag"] = "diag",
        approximation_accuracy: Literal["exact", "approx"] = "exact",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        assert hessian_shape in ("full", "block", "diag")  # TODO: better name
        assert approximation_accuracy in ("exact", "approx")  # TODO: better name

        self.hessian_shape = hessian_shape
        self.approximation_accuracy = approximation_accuracy

    @abstractmethod
    @torch.no_grad()
    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        model: nnj.Sequential,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def compute_gradient(
        self, x: torch.Tensor, target: torch.Tensor, model: nnj.Sequential, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def compute_hessian(
        self, x: torch.Tensor, target: Optional[torch.Tensor], model: nnj.Sequential, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
