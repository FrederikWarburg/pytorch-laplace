# Mean Squared Error

The Gaussian ***loss*** $\mathcal{L}^{\texttt{G}}$ for a target $y$ is a function of $z$ defined by (for a fixed isotropic covariance matrix $\Sigma=\sigma^2\mathbb{I}$)
```math
\begin{align}
    \mathcal{L}^{\texttt{G}}_y(z)
    & := -\log \mathbb{P}(y|\mathcal{N}(z,\Sigma)) \\ 
    & = \frac{1}{2}(y-z)^\top \Sigma^{-1} (y-z) + \log(\sqrt{2\pi\det\Sigma}) \\
    & = \frac{1}{2\sigma^2} \|y-z\|^2 + \text{const}
\end{align}
```
which non-Bayesian folks refer to as Mean Squared Error (MSE) or Sum of Squared Error (SSE).

The ***gradient*** of this loss with respect to the NN output $z$ is
```math
\begin{equation}
    \nabla_z \mathcal{L}^{\texttt{G}}_y(z)
    =
    \sigma^{-2} (y-z)
\end{equation}
```

The ***hessian*** of this loss with respect to the NN output $z$ is
```math
\begin{equation}
    \nabla^2_z \mathcal{L}^{\texttt{G}}_y(z)
    =
    \sigma^{-2} \mathbb{I}
\end{equation}
```

Note that this hessian is independent on the target, and thus the Generalized Gauss Newton (GGN) is equal to the Fisher matrix. This is also implied by the fact that $\mathcal{N}$ is an exponential distribution and we used the natural parameter $z$.

### Ok, but now use it with a Neural Network
Let $f_\theta(x)=z$ be a Neural Network with parameter $\theta$ that maps $x\rightarrow z$.
We minimize the loss (aka. maximize the gaussian log-likelihood)
```math
\begin{equation}
    \min_\theta \mathcal{L}^{\texttt{G}}_x(f_\theta(x))
\end{equation}
```

Zero, first and second order derivatives (with respect to $\theta$) can be computed by
```python
from stochman.hessian import MSEHessianCalculator

mse = MSEHessianCalculator(wrt="weight", shape="diagonal", speed="half")

log_gaussian = mse.compute_loss(imgs, imgs, model)
gradient_log_gaussian = mse.compute_gradient(imgs, imgs, model) 
hessian_log_gaussian = mse.compute_hessian(imgs, model)
```
