from typing import Tuple, List

import nnj
import torch
from torch.nn.utils import parameters_to_vector

from pytorch_laplace.laplace.base import BaseLaplace


class BlockLaplace(BaseLaplace):
    def laplace(
        self,
        x: torch.Tensor,
        hessian: torch.Tensor,
        model: nnj.Sequential,
        scale: float = 1,
        prior_prec: float = 1,
        n_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the Laplace approximation of the posterior distribution of the parameters.

        Args:
            x: The input data.
            y: The target data.
            model: The neural network.
            prior_prec: The precision of the prior distribution.
            n_samples: The number of samples to draw.
            scale: The scale of the posterior distribution.
            device

        """

        sigma_q = self.posterior_scale(hessian=hessian, scale=scale, prior_prec=prior_prec)
        mu_q = parameters_to_vector(model.parameters())

        samples = self.sample_from_normal(mu_q, sigma_q, n_samples)
        pred_mu, pred_sigma = self.normal_from_samples(x, samples, model)

        return pred_mu, pred_sigma

    def linearized_laplace(
        self,
        x: torch.Tensor,
        hessian: torch.Tensor,
        model: nnj.Sequential,
        scale: float = 1,
        prior_prec: float = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the linearized Laplace approximation of the posterior distribution of the parameters.

        .. note::
            This does not require sampling from the posterior distribution with using nnj as backend!

        Args:
            x: The input data.
            hessian: The Hessian of the loss function.
            model: The neural network.
            scale: The scale of the posterior distribution.
            prior_prec: The precision of the prior distribution.

        """

        raise NotImplementedError

    def sample_from_normal(self, mu: torch.Tensor, scale: List[torch.Tensor], n_samples: int = 100):
        """
        Sample parameters from the posterior distribution of the parameters.

        .. math::
            samples = parameters + posterior\_scale * epsilon

        Args:
            mu: The parameters of the model.
            scale: The posterior scale of the parameters.
            n_samples: The number of samples to draw.
        """

        n_samples = torch.tensor([n_samples])
        count = 0
        param_samples = []
        for post_scale_layer in scale:
            n_param_layer = len(post_scale_layer)

            layer_param = mu[count : count + n_param_layer]
            normal = torch.distributions.multivariate_normal.MultivariateNormal(
                layer_param, covariance_matrix=post_scale_layer
            )
            samples = normal.sample(n_samples)
            param_samples.append(samples)

            count += n_param_layer

        param_samples = torch.cat(param_samples, dim=1).to(mu.device)
        return param_samples

    @torch.no_grad()
    def normal_from_samples(
        self,
        x: torch.Tensor,
        samples: torch.Tensor,
        model: nnj.Sequential,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the emperical mean and scale of the normal distribution from samples

        Args:
            x: The input data
            samples: The samples
            model: The neural network.
        """

        raise NotImplementedError

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
