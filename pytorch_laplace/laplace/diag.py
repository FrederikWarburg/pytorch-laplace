import re
from typing import Optional, Tuple

import nnj
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from pytorch_laplace.laplace.base import BaseLaplace


class DiagLaplace(BaseLaplace):
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
            hessian: The Hessian of the loss function.
            model: The neural network.
            prior_prec: The precision of the prior distribution.
            n_samples: The number of samples to draw.
            scale: The scale of the posterior distribution.
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

        sigma_q = self.posterior_scale(hessian=hessian, scale=scale, prior_prec=prior_prec)

        with torch.no_grad():
            # forward the mode
            pred_mu = model(x)

            # forward the covariance
            pred_sigma = model.jmjTp(
                x,
                None,
                sigma_q.unsqueeze(0).expand(x.shape[0], -1),
                wrt="weight",
                from_diag=True,
                to_diag=True,
            )

        pred_sigma = torch.sum(pred_sigma, dim=-1)

        return pred_mu, pred_sigma

    def sample_from_normal(
        self,
        mu: torch.Tensor,
        scale: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """
        Sample parameters from normal distribution

        .. math::
            samples = mu + scale * epsilon

        Args:
            mu: The parameters of the model.
            scale: The posterior scale of the parameters.
            n_samples: The number of samples to draw.
        """

        n_params = len(mu)
        samples = torch.randn(n_samples, n_params, device=mu.device)
        samples = samples * scale.view(1, n_params)
        return mu.view(1, n_params) + samples

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

        mu_params = parameters_to_vector(model.parameters())

        pred_mu = 0
        pred_mu2 = 0
        for sample in samples:
            vector_to_parameters(sample, model.parameters())

            with torch.no_grad():
                pred = model(x)

            pred_mu += pred
            pred_mu2 += pred**2

        pred_mu /= len(samples)
        pred_mu2 /= len(samples)

        pred_sigma = pred_mu2 - pred_mu**2

        vector_to_parameters(mu_params, model.parameters())

        return pred_mu, pred_sigma

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
