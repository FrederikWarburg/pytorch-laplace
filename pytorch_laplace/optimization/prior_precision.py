import torch


def log_det_ratio(hessian: torch.Tensor, prior_prec: torch.Tensor):
    posterior_precision = hessian + prior_prec
    log_det_prior_precision = len(hessian) * prior_prec.log()
    log_det_posterior_precision = posterior_precision.log().sum()
    return log_det_posterior_precision - log_det_prior_precision


def scatter(mu_q: torch.Tensor, prior_precision_diag: torch.Tensor):
    return (mu_q * prior_precision_diag) @ mu_q


def log_marginal_likelihood(mu_q: torch.Tensor, hessian: torch.Tensor, prior_prec: torch.Tensor):
    # we ignore neg log likelihood as it is constant wrt prior_prec
    neg_log_marglik = -0.5 * (log_det_ratio(hessian, prior_prec) + scatter(mu_q, prior_prec))
    return neg_log_marglik


def optimize_prior_precision(
    mu_q: torch.Tensor, hessian: torch.Tensor, prior_prec: torch.Tensor, n_steps: int = 100
):
    log_prior_prec = prior_prec.log()
    log_prior_prec.requires_grad = True
    optimizer = torch.optim.Adam([log_prior_prec], lr=1e-1)
    for _ in range(n_steps):
        optimizer.zero_grad()
        prior_prec = log_prior_prec.exp()
        neg_log_marglik = -log_marginal_likelihood(mu_q, hessian, prior_prec)
        neg_log_marglik.backward()
        optimizer.step()

    prior_prec = log_prior_prec.detach().exp()

    return prior_prec
