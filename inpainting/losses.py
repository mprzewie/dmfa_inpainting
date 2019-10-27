from typing import Callable

import torch
from torch.distributions import MultivariateNormal


def nll_batch_loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        sample_loss: Callable[[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ], torch.Tensor]
) -> torch.Tensor:
    return torch.stack([
        sample_loss(x, j, p, m, a, d)
        for (x, j, p, m, a, d) in zip(X, J, P, M, A, D)
    ])

def nll_masked_sample_loss(
        x: torch.Tensor,
        j: torch.Tensor,
        p: torch.Tensor,
        m: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor
) -> torch.Tensor:
    mask_inds = (j == 0).nonzero().squeeze()
    x_masked = torch.index_select(x, 0, mask_inds)
    a_masked = torch.index_select(a, 2, mask_inds)
    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    losses_for_p = []
    for (p_i, m_i, d_i, a_i) in zip(p, m_masked, d_masked, a_masked):
        if a.shape[1] < 0:
            losses_for_p.append(torch.tensor(0.0, requires_grad=False))
        cov = (a_i.T @ a_i) + torch.diag(d_i ** 2)
        mvn_d = MultivariateNormal(m_i, cov)  # calculate this manually
        losses_for_p.append(- mvn_d.log_prob(x_masked))
    return torch.stack(losses_for_p).sum()
