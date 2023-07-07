from abc import ABC, abstractmethod
from typing import Literal

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

        assert shape in ("full", "block", "diag")  # TODO: better name
        assert speed in ("half", "fast")  # TODO: better name

        self.shape = shape
        self.speed = speed

    @abstractmethod
    @torch.no_grad()
    def compute_loss(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        nnj_module: nnj.Sequential,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def compute_gradient(
        self, x: torch.Tensor, target: torch.Tensor, nnj_module: nnj.Sequential, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def compute_hessian(
        self, x: torch.Tensor, target: torch.Tensor, nnj_module: nnj.Sequential, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
