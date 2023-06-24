import torch
from torch.nn.utils import parameters_to_vector

from pytorch_laplace.laplace.base import BaseLaplace


class DiagLaplace(BaseLaplace):
    def sample(self, parameters, posterior_scale, n_samples=100):
        n_params = len(parameters)
        samples = torch.randn(n_samples, n_params, device=parameters.device)
        samples = samples * posterior_scale.view(1, n_params)
        return parameters.view(1, n_params) + samples

    def posterior_scale(self, hessian, scale=1, prior_prec=1):
        posterior_precision = hessian * scale + prior_prec
        return 1.0 / (posterior_precision.sqrt() + 1e-6)

    def init_hessian(self, data_size, net, device):
        hessian = data_size * torch.ones_like(parameters_to_vector(net.parameters()), device=device)
        return hessian

    def scale(self, h_s, b, data_size):
        return h_s / b * data_size

    def average_hessian_samples(self, hessian, constant):
        # average over samples
        hessian = torch.stack(hessian).mean(dim=0) if len(hessian) > 1 else hessian[0]

        # get posterior_precision
        return constant * hessian
