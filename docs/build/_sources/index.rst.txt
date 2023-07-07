Pytorch-laplace Documentation
===================================

Pytorch-laplace provides a simple API for Laplace approximation (LA) in PyTorch. With minimal code changes, you can use it to approximate the posterior of any PyTorch model.
It supports both Laplace and Linearized Laplace. It uses nnj as backend for approximate hessian computations, which is an order of magnitude faster and more memory efficient than alternatives.
The repo focuses on diagonal hessian approximations, but also supports low-rank hessian approximations such as the kronicker factorization.

Github: https://github.com/FrederikWarburg/pytorch-laplace

Authors: Frederik Warburg and Marco Miani

Installation
===============================

**Dependence**: Please install Pytorch first.

The easiest way is to install from PyPI:

.. code-block:: console

   $ pip install pytorch-laplace

Or install from source:

.. code-block:: console

   $ git clone https://github.com/FrederikWarburg/pytorch-laplace
   $ cd pytorch-laplace
   $ pip install -e .



Want to learn more about the Laplace approximation?
==============================================================


Check out our educational content, e.g. start reading our :ref:`Introduction to Laplace<introduction>`  to learn more about the Laplace approximation.
The document seek to provide a simple introduction to the Laplace approximation and how to use it in PyTorch.

.. toctree::
   :glob:
   :hidden:
   :caption: Learn about Laplace 

   laplace/*
   

Usage
===============================


.. code-block:: python

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


   # fit hessian
   hessian = fit_hessian(network_nnj, train_loader, device)

   # sample from the posterior
   pred_mu, pred_sigma = sample_laplace(network_nnj, hessian, test_loader, device, n_samples=100)


Links:
-------------

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Python API

   apis/*
