import torch
from torch.nn.utils import parameters_to_vector

from pytorch_laplace.laplace.base import BaseLaplace


class BlockLaplace(BaseLaplace):
    def sample(self, parameters, posterior_scale, n_samples=100):
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

    def posterior_scale(self, hessian, scale=1, prior_prec=1):

        posterior_precision = [
            h * scale + torch.diag_embed(prior_prec * torch.ones(h.shape[0])) for h in hessian
        ]
        posterior_scale = [torch.cholesky_inverse(layer_post_prec) for layer_post_prec in posterior_precision]
        return posterior_scale

    def init_hessian(self, data_size, net, device):

        hessian = []
        for layer in net:
            # if parametric layer
            if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
                params = parameters_to_vector(layer.parameters())
                n_params = len(params)
                hessian.append(data_size * torch.ones(n_params, n_params, device=device))

        return hessian

    def scale(self, h_s, b, data_size):
        return [h / b * data_size for h in h_s]

    def aveage_hessian_samples(self, hessian, constant):
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