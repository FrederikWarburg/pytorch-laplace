# pytorch-laplace

Pytorch-laplace provides a simple API for Laplace approximation (LA) in PyTorch. With minimal code changes, you can use it to approximate the posterior of any PyTorch model.
It supports both Laplace and Linearized Laplace. It allows using nnj as backend for approximate hessian computations, which is an order of magnitude faster and more memory efficient than alternatives.
The repo focuses on diagonal hessian approximations, but also supports low-rank hessian approximations such as the kronicker factorization.

Github: https://github.com/FrederikWarburg/pytorch-laplace
Documentation: https://frederikwarburg.github.io/pytorch-laplace/
Authors: Frederik Warburg and Marco Miani

# Installation

**Dependence**: Please install Pytorch first.

The easiest way is to install from PyPI:

```bash
pip install pytorch-laplace
```
Or install from source:

```bash
git clone https://github.com/FrederikWarburg/pytorch-laplace
cd pytorch-laplace
pip install -e .
```

# Usage

```python
import torch.nn as nn
import nnj

# Import the laplace approximation
from pytorch_laplace import MSEHessianCalculator
from pytorch_laplace.laplace.diag import DiagLaplace
from pytorch_laplace.optimization.prior_precision import optimize_prior_precision

# Define you sequential model
network_nn = torch.nn.Sequential(
   nn.Linear(),
   nn.Tanh(),
   nn.Linear(),
)

# convert to nnj
network_nnj = nnj.utils.convert_to_nnj(
   network_nn,
)


 # define HessianCalculator
 # this class assumes you are using MSE loss
 # to train your network.
 hessian_calculator = nnj.MSEHessianCalculator(
   hessian_shape="diag", # uses a diagonal approximation of the GGN
   approximation_accuracy="exact", # alternatively choose "approx", which scales linearly with the output dimension, rather than quadratically,
   backend="nnj",
 )

 # initialize hessian
 hessian = torch.zeros_like(model.parameters())
 for x, y in train_loader:
     # compute hessian approximation
     hessian += hessian_calculator.compute_hessian(
       x=x, model=model,
     )

 # Compute the posterior
 sampler = DiagLaplace(backend="nnj")

 for x, y in test_loader:

     # get predictive distribution
     pred_mu, pred_sigma = sampler.laplace(
         x=x,
         model=model,
         hessian=hessian,
         prior_precision=1.0,
         scale=1.0,
         num_samples=100,
     )
```

## BibTeX
If you want to cite the framework:

```bibtex
@article{software:pytorch-laplace,
  title={Pytorch-laplace},
  author={Frederik Warburg and Marco Miani},
  journal={GitHub. Note: [https://github.com/FrederikWarburg/pytorch-laplace](https://github.com/FrederikWarburg/pytorch-laplace)},
  year={2023}
}
```
