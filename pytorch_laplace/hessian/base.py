from abc import ABC
from typing import Literal

from torch import nn


class HessianCalculator(ABC, nn.Module):
    def __init__(
        self,
        wrt: Literal = "weight",
        shape: Literal = "diagonal",
        speed: Literal = "half",
        method: Literal = "",
    ) -> None:
        super().__init__()

        assert wrt in ("weight", "input")
        assert shape in ("full", "block", "diagonal")
        assert speed in ("slow", "half", "fast")
        assert method in ("", "full", "pos", "fix")

        self.wrt = wrt
        self.shape = shape
        self.speed = speed
        self.method = method
        if speed == "slow":
            # second order
            raise NotImplementedError

    def compute_loss(self, x, target, nnj_module, tuple_indices=None):
        raise NotImplementedError

    def compute_gradient(self, x, target, nnj_module, tuple_indices=None):
        raise NotImplementedError

    def compute_hessian(self, x, nnj_module, tuple_indices=None):
        raise NotImplementedError
