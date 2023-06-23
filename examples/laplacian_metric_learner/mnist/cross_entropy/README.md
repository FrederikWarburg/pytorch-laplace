# (Binary) Cross Entropy

The ***Binary Cross-Entropy*** loss $\mathcal{L}^{\texttt{BCE}}$ for a target $y$ is a function of $z$ defined by
```math
\begin{align}
    \mathcal{L}^{\texttt{BCE}}_y({z})
    & := -\log \mathbb{P}
    \left(
        y\Big|\text{Bernoulli}
            \left(
                \frac{e^{{z}}}{1+e^{{z}}}, \frac{1}{1+e^{{z}}}
            \right)
    \right) \\ 
    & = -\left(
            {y} \cdot \log
            \left(
                \frac{e^{{z}}}{1+e^{{z}}}
            \right)
            + (1-{y}) \cdot \log 
            \left(
                \frac{1}{1+e^{{z}}}
            \right)
        \right) \\ 
    & = -\sum_p 
        \left(
            {y}_p \cdot \log
            \left(
                \frac{e^{{z}_p}}{1+e^{{z}_p}}
            \right)
            + (1-{y}_p) \cdot \log 
            \left(
                \frac{1}{1+e^{{z}_p}}
            \right)
        \right) \\ 
    & = -\sum_p
        \left(
            {y}_p \cdot \log(e^{{z}_p})
            + (1-{y}_p) \cdot \log(1)
            - ({y}_p + 1-\bar{y}_p) \cdot \log(1+e^{{z}_p})
        \right) \\ 
    & = -\sum_p 
        \left(
            {y}_p \cdot {z}_p
            - \log(1+e^{{z}_p})
        \right)
\end{align}
```

The ***gradient*** of this loss with respect to the Neural Network output $z$ is
```math
\begin{align}
    \nabla_z \mathcal{L}^{\texttt{BCE}}_y(z)
    & = 
    \frac{e^{{z}}}{1+e^{{z}}}
    - {y}
\end{align}
```

The ***hessian*** of this loss with respect to the Neural Network output $z$ is
```math
\begin{align}
    \nabla^2_{\bar{z}} \mathcal{L}^{\texttt{BCE}}_y({z})
    & = \sum_p \nabla^2_{{z}} \log(1+e^{{z}_p}) \\
    & = \texttt{diag}_p 
        \left(
            \frac{e^{{z}_p}}{1+e^{{z}_p}}
            -
            \left(
            \frac{e^{{z}_p}}{1+e^{{z}_p}}
            \right)^2
        \right)
\end{align}
```

Note that this hessian is independent on the target, and thus the Generalized Gauss Newton (GGN) is equal to the Fisher matrix. This is also implied by the fact that Bernoulli is an exponential distribution and we used the natural parameter $z$.

### Ok, but now use it with a Neural Network
Let $f_\theta(x)=z$ be a Neural Network with parameter $\theta$ that maps $x\rightarrow z$.
We minimize the loss (aka. maximize the bernoulli log-likelihood)
```math
\begin{equation}
    \min_\theta \mathcal{L}^{\texttt{BCE}}_x(f_\theta(x))
\end{equation}
```

Zero, first and second order derivatives (with respect to $\theta$) can be computed by
```python
from stochman.hessian import BCEHessianCalculator

cross_entropy = BCEHessianCalculator(wrt="weight", shape="diagonal", speed="half")

log_bernoulli = cross_entropy.compute_loss(imgs, imgs, model)
gradient_log_bernoulli = cross_entropy.compute_gradient(imgs, imgs, model) 
hessian_log_bernoulli = cross_entropy.compute_hessian(imgs, model)
