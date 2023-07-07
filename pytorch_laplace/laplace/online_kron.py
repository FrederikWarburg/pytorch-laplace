from typing import List, Optional, Tuple, Union

import nnj
import torch
from torch.nn.utils import parameters_to_vector

from pytorch_laplace.laplace.diag import DiagLaplace


class OnlineBlockLaplace(DiagLaplace):
    def init_hessian(self, data_size: int, net: nnj.Sequential, device: str):
        """
        Initialize the low-rank Hessian matrix.

        Args:
            data_size: The size of the dataset.
            net: The neural network.
            device: The device to use.
        """
        hessian = []
        for layer in net:
            # if parametric layer
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                params = parameters_to_vector(layer.parameters())
                n_params = len(params)
                hessian.append(data_size * torch.ones(n_params, n_params, device=device))

        return hessian

    def scale(self, hessian_batch: torch.Tensor, batch_size: int, data_size: int):
        """
        Scales the Hessian approximated from a batch to the full dataset.

        Args:
            hessian_batch: The Hessian approximated from a batch.
            batch_size: The size of the batch.
            data_size: The size of the dataset.
        """
        return [h / batch_size * data_size for h in hessian_batch]

    def average_hessian_samples(self, hessian: Union[torch.Tensor, List], constant: Optional[float] = 1):
        """
        Average the Hessian samples.

        Args:
            hessian: The Hessian samples.
            constant: The constant to multiply the Hessian with.
        """
        n_samples = len(hessian)
        n_layers = len(hessian[0])
        hessian_mean = []
        for i in range(n_layers):
            tmp = None
            for s in range(n_samples):
                if tmp is None:
                    tmp = hessian[s][i]
                else:
                    tmp += hessian[s][i]

            tmp = tmp / n_samples
            tmp = constant * tmp + torch.diag_embed(torch.ones(len(tmp), device=tmp.device))
            hessian_mean.append(tmp)

        return hessian_mean
