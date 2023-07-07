from typing import List, Optional, Union

import nnj
import torch
from torch.nn.utils import parameters_to_vector

from pytorch_laplace.laplace.diag import DiagLaplace


class OnlineDiagLaplace(DiagLaplace):
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

    def average_hessian_samples(self, hessian: Union[torch.Tensor, List], constant: Optional[float] = 1):
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
