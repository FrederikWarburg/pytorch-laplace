from typing import Optional

import nnj
import torch
from torch.nn.utils import parameters_to_vector

from pytorch_laplace.laplace.base import BaseLaplace


class BlockLaplace(BaseLaplace):
    def sample(self, parameters: torch.Tensor, posterior_scale: float, n_samples: int = 100):
        """
        Sample parameters from the posterior distribution of the parameters.

        .. math::
            samples = parameters + posterior\_scale * epsilon

        Args:
            parameters: The parameters of the model.
            posterior_scale: The posterior scale of the parameters.
            n_samples: The number of samples to draw.
        """

        n_samples = torch.tensor([n_samples])
        count = 0
        param_samples = []
        for post_scale_layer in posterior_scale:
            n_param_layer = len(post_scale_layer)

            layer_param = parameters[count : count + n_param_layer]
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                layer_param, covariance_matrix=post_scale_layer
            )
            samples = normal.sample(n_samples)
            param_samples.append(samples)

            count += n_param_layer

        param_samples = torch.cat(param_samples, dim=1).to(parameters.device)
        return param_samples

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
        posterior_precision = [
            h * scale + torch.diag_embed(prior_prec * torch.ones(h.shape[0])) for h in hessian
        ]
        posterior_scale = [torch.cholesky_inverse(layer_post_prec) for layer_post_prec in posterior_precision]
        return posterior_scale

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

    def aveage_hessian_samples(self, hessian: torch.Tensor, constant: Optional[float] = 1):
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
