.. _introduction:

Introduction
===================================

The Laplace approximation (McKay, 1992) is a method to quantify the uncertainty of a neural network.
It is based on the assumption that the posterior distribution of the weights of a neural network is Gaussian.
The Laplace approximation is a fast and simple method to approximate the posterior distribution of the weights of a neural network.
In this document, we will explain the Laplace approximation and how to use it in the context of Bayesian neural networks.


Intuition
===================================

.. video:: ../_static/images/laplace-intuition.mp4
  :alt: Laplace approximation intuition
  :width: 800
  :nocontrols:
  :loop:
  :autoplay: 
  :muted:
  

The figure above shows the loss (or negative log likelihood) for two parameters in a neural network.
After training, the parameters have converged to a local minimum of the loss, illustrated by the blue star :math:`\theta^*`.
If a parameter is in a steep valley, changing it a little bit will increase the loss a lot.
This means that we are fairly certain about the specific value of the parameter. On the other hand, 
if a paramter is in a flat valley, changing it a little bit will not increase the loss a lot. Thus,
we are uncertain about the specific value of the parameter. 

The steepness of the valley is determined by the second derivative of the loss, also called the Hessian.
Thus, the inverse of the Hessian determines the uncertainty of the parameters. In the following,
we will derive the Laplace approximation and show that it correspond to assuming a Gaussian distribution 
over the parameters (Gaussian weight-posterior).


Derivations
===================================

Laplace approximations (LA) can be applied for every
loss function :math: `\mathcal{L}` that can be interpreted as an unnormalized
log-posterior by performing a second-order Taylor expansion around a chosen weight vector :math:`\theta^*`:

.. math::
    \log p(\boldsymbol{\theta} \mid \mathcal{D})=\mathcal{L}^*+\left(\boldsymbol{\theta}-\boldsymbol{\theta}^*\right)^{\top} \nabla \mathcal{L}^*+\frac{1}{2}\left(\boldsymbol{\theta}-\boldsymbol{\theta}^*\right)^{\top} \nabla^2 \mathcal{L}^*\left(\boldsymbol{\theta}-\boldsymbol{\theta}^*\right)+\mathcal{O}\left(\left\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\right\|^3\right)


Imposing the unnormalized log-posterior to be a second-order polynomial is equivalent to 
assuming the posterior to be Gaussian.

If :math:`\theta^*` is the maximum a posteriori (MAP) estimate, a common assumption is that the gradient is zero, 
such that the first-order term in the Taylor expansion vanishes, and the Taylor expansion simplifies to:

.. math::
    \begin{aligned}
    \mathcal{L}(\theta) \approx & \mathcal{L}^*+\frac{1}{2}\left(\theta-\theta^*\right)^{\top} \nabla^2 \mathcal{L}^*\left(\theta-\theta^*\right)
    \end{aligned}

For common loss functions, such as the MSE and Cross-Entropy loss, the Hessian is positive definite,
and thus, we can write the weight posterior as:

.. math::
    \begin{aligned}
    p(\theta \mid \mathcal{D})=\mathcal{N}\left(\theta \mid \theta^*,\left(\nabla_\theta^2 \mathcal{L}_{\text {con }}\left(\theta^* ; \mathcal{D}\right)+\sigma_{\text {prior }}^{-2} \mathbb{I}\right)^{-1}\right)
    \end{aligned}

where :math:`\mathcal{N}` denotes the Gaussian distribution.

Hessian approximations
===================================

The main challenge of the Laplace approximation is to compute the Hessian of 
the neural network with respect to the parameters :math:`\nabla_\theta^2` and inverting this, 
as it is very computationally expensive. Therefore, there has been proposed several
approximations to the Hessian. In the following, we will describe the most common.

The Hessian is commonly approximiation with the **Generalized Gauss-Newton** approximation:

.. math::
    \nabla_{\boldsymbol{\theta}^{(l)}}^2 \mathcal{L}\left(f_{\boldsymbol{\theta}}(\boldsymbol{x})\right) \approx J_{\boldsymbol{\theta}^{(l)}} f_{\boldsymbol{\theta}}(\boldsymbol{x})^{\top} \cdot \nabla_{\boldsymbol{\hat{y}}}^2 \mathcal{L}\left(\boldsymbol{\hat{y}}\right) \cdot J_{\boldsymbol{\theta}^{(l)}} f_{\boldsymbol{\theta}}(\boldsymbol{x}),

for a single layer :math:`l`, which neglects second order derivatives of :math:`f` w.r.t. the parameters.
Besides, the computational benefits of this approximations, the GGN also ensures that the Hessian is semi-positive definite.

Another common approximation is the **diagonal** approximation, which is the diagonal of the Hessian.
This ensures that the models scales linearly with the number of parameters, rather than quadratically.

There exists several pytorch backends that supports, efficient computation of the Hessian, such as

* NNJ (which efficiently implements jacobian-vector and jacobina-matrix products)
* Backpack (which extends pytorch autograd with (approximate) second order derivatives)
* ASDL (which is similar to Backpack)


This repo supports all backends, but focuses on the NNJ, because it is an order of magnitude faster and more memory efficient
than the other backends. NNJ does not replies on autograd, but instead uses a custom PyTorch jacobian-vector products,
which are more efficient for approximate second-order methods. Furthermore, NNJ allows to scale the laplace approximation
to large neural networks with large input, which is not possible with the other backends. NNJ also allows to compute linearized laplace
without relying on samples (single forward pass). The main disadvantage of NNJ is that it requires the user to implement jacobian of layers.

We can get the hessian of the loss w.r.t. the parameters as follows:

.. code-block:: python

    # define data loader
    train_loader = torch.utils.data.DataLoader()

    # define model (convert torch.Sequential to nnj.Sequential)
    model = nnj.convert_to_nnj(model)

    # define HessianCalculator
    hessian_calculator = nnj.MSEHessianCalculator(
      approximation="diagonal", # better name
      approximation2="diagonal", # better name
    )

    # initialize hessian
    hessian = torch.zeros_like(model.parameters())
    for x, y in train_loader:
        # compute hessian
        hessian += hessian_calculator(
          x=x, 
          y=y, 
          nnj_module=model
        )

Sampling (Laplace)
===================================

Once, we have a weight-posterior (a distribution of weights), we can sample from it to obtain a distribution of predictions.

.. math::
    \begin{aligned}
    p(y \mid \boldsymbol{x}, \mathcal{D}) &=\int p(y \mid \boldsymbol{x}, \boldsymbol{\theta}) p(\boldsymbol{\theta} \mid \mathcal{D}) d \boldsymbol{\theta} \\
    & \approx \frac{1}{K} \sum_{k=1}^{K} p(y \mid \boldsymbol{x}, \boldsymbol{\theta}^{(k)}), \quad \boldsymbol{\theta}^{(k)} \sim p(\boldsymbol{\theta} \mid \mathcal{D})
    \end{aligned}

where :math:`K` is the number of samples. In code, this can be implemented as follows:

.. code-block:: python

    # define dataloader
    test_loader = torch.utils.data.DataLoader()

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
            device="cuda:0",
        )




Sampling (Linearized Laplace)
===================================




Getting Started
===================================