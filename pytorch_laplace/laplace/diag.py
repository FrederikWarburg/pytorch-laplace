from typing import Optional

import nnj
import torch
from torch.nn.utils import parameters_to_vector

from pytorch_laplace.laplace.base import BaseLaplace


class DiagLaplace(BaseLaplace):
    def sample(
        self,
        parameters: torch.Tensor,
        posterior_scale: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """
        Sample parameters from the posterior distribution of the parameters.

        .. math::
            samples = parameters + posterior\_scale * epsilon

        Args:
            parameters: The parameters of the model.
            posterior_scale: The posterior scale of the parameters.
            n_samples: The number of samples to draw.
        """

        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device=parameters.device)
        samples = samples * posterior_scale.view(1, n_params)
        return parameters.view(1, n_params) + samples

    def posterior_scale(self, hessian: torch.Tensor, scale: float = 1, prior_prec: float = 1):
        """
        Compute the posterior scale of the parameters.

        .. math::
            posterior\_scale = 1 / \sqrt{hessian * scale + prior\_prec}

        Args:
            hessian: The Hessian of the loss function.
            scale: The scale of the posterior distribution.
            prior_prec: The precision of the prior distribution.
        """
        posterior_precision = hessian * scale + prior_prec
        return 1.0 / (posterior_precision.sqrt() + 1e-6)

    def init_hessian(self, data_size: int, net: nnj.Sequential, device: str):
        """
        Initialize the diagonal Hessian matrix.

        Args:
            data_size: The size of the dataset.
            net: The neural network.
            device: The device to use.
        """
        hessian = data_size * torch.ones_like(parameters_to_vector(net.parameters()), device=device)
        return hessian

    def scale(self, hessian_batch: torch.Tensor, batch_size: int, data_size: int):
        """
        Scales the Hessian approximated from a batch to the full dataset.

        Args:
            hessian_batch: The Hessian approximated from a batch.
            batch_size: The size of the batch.
            data_size: The size of the dataset.
        """

        return hessian_batch / batch_size * data_size

    def average_hessian_samples(self, hessian: torch.Tensor, constant: Optional[float] = 1):
        """
        Average over the Hessian samples.

        Args:
            hessian: The Hessian samples.
            constant: The constant to multiply the Hessian with.
        """

        # average over samples
        hessian = torch.stack(hessian).mean(dim=0) if len(hessian) > 1 else hessian[0]

        # get posterior_precision
        return constant * hessian
