import torch

from pytorch_laplace.hessian.base import HessianCalculator


class BCEHessianCalculator(HessianCalculator):
    "Binary Cross Entropy"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.method == ""

    def compute_loss(self, x, target, nnj_module, tuple_indices=None):
        with torch.no_grad():
            val = nnj_module(x)
            assert val.shape == target.shape

            bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
            cross_entropy = -(target * torch.log(bernoulli_p) + (1 - target) * torch.log(1 - bernoulli_p))

            # average along batch size
            cross_entropy = torch.mean(cross_entropy, dim=0)
            # sum along other dimensions
            cross_entropy = torch.sum(cross_entropy)
            return cross_entropy

    def compute_gradient(self, x, target, nnj_module, tuple_indices=None):
        with torch.no_grad():
            val = nnj_module(x)
            assert val.shape == target.shape

            bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
            gradient = bernoulli_p - target

            # backpropagate through the network
            gradient = gradient.reshape(val.shape[0], -1)
            gradient = nnj_module._vjp(x, val, gradient, wrt=self.wrt)

            # average along batch size
            gradient = torch.mean(gradient, dim=0)
            return gradient

    def compute_hessian(self, x, nnj_module, tuple_indices=None):
        with torch.no_grad():
            val = nnj_module(x)

            bernoulli_p = torch.exp(val) / (1 + torch.exp(val))
            H = bernoulli_p - bernoulli_p**2
            H = H.reshape(val.shape[0], -1)  # hessian in diagonal form

            # backpropagate through the network
            Jt_H_J = nnj_module._jTmjp(
                x,
                val,
                H,
                wrt=self.wrt,
                from_diag=True,
                to_diag=self.shape == "diagonal",
                diag_backprop=self.speed == "fast",
            )
            # average along batch size
            Jt_H_J = torch.mean(Jt_H_J, dim=0)
            return Jt_H_J
