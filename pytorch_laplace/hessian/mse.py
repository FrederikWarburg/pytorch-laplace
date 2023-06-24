import torch

from pytorch_laplace.hessian.base import HessianCalculator


class MSEHessianCalculator(HessianCalculator):
    "Mean Square Error"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.method == ""

    def compute_loss(self, x, target, nnj_module, tuple_indices=None):
        with torch.no_grad():
            val = nnj_module(x)
            assert val.shape == target.shape

            # compute Gaussian log-likelihood
            mse = 0.5 * (val - target) ** 2

            # average along batch size
            mse = torch.mean(mse, dim=0)

            # sum along other dimensions
            mse = torch.sum(mse)

            return mse

    def compute_gradient(self, x, target, nnj_module, tuple_indices=None):
        with torch.no_grad():
            val = nnj_module(x)
            assert val.shape == target.shape

            # compute gradient of the Gaussian log-likelihood
            gradient = val - target

            # backpropagate through the network
            gradient = gradient.reshape(val.shape[0], -1)
            gradient = nnj_module._vjp(x, val, gradient, wrt=self.wrt)

            # average along batch size
            gradient = torch.mean(gradient, dim=0)
            return gradient

    def compute_hessian(self, x, nnj_module, tuple_indices=None):
        # compute Jacobian sandwich of the identity for each element in the batch
        # H = identity matrix (None is interpreted as identity by jTmjp)

        with torch.no_grad():
            val = nnj_module(x)

            # backpropagate through the network
            Jt_J = nnj_module._jTmjp(
                x,
                val,
                None,
                wrt=self.wrt,
                to_diag=self.shape == "diagonal",
                diag_backprop=self.speed == "fast",
            )
            # average along batch size
            Jt_J = torch.mean(Jt_J, dim=0)
            return Jt_J
