from typing import Callable

import torch
from torch.distributions import MultivariateNormal

InpainterLossFn = Callable[[
                               torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               torch.Tensor,
                               torch.Tensor
                           ],
                           torch.Tensor
]


def nll_masked_sample_loss_v1(
        x: torch.Tensor,
        j: torch.Tensor,
        p: torch.Tensor,
        m: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor
) -> torch.Tensor:
    """A very unvectorized loss"""
    mask_inds = (j == 0).nonzero().squeeze()
    x_masked = torch.index_select(x, 0, mask_inds)
    a_masked = torch.index_select(a, 2, mask_inds)
    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    covs = torch.bmm(a_masked.transpose(1, 2), a_masked)
    losses_for_p = []
    for (p_i, m_i, d_i, cov_i) in zip(p, m_masked, d_masked, covs):
        if cov_i.shape[1] > 0:
            cov = cov_i + torch.diag(d_i ** 2)
            mvn_d = MultivariateNormal(m_i, cov)  # calculate this manually
            losses_for_p.append(- mvn_d.log_prob(x_masked.float()))
    return torch.stack(losses_for_p).sum()


def inpainter_batch_loss_fn(
        sample_loss: Callable[[
                                  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
                              ], torch.Tensor] = nll_masked_sample_loss_v1) -> InpainterLossFn:
    def loss(
            X: torch.Tensor,
            J: torch.Tensor,
            P: torch.Tensor,
            M: torch.Tensor,
            A: torch.Tensor,
            D: torch.Tensor,
    ) -> torch.Tensor:
        return torch.stack([
            sample_loss(x, j, p, m, a, d)
            for (x, j, p, m, a, d) in zip(X, J, P, M, A, D)
        ]).mean()

    return loss


def r2_total_batch_loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
) -> torch.Tensor:
    return ((X - M[:, 0, :]) ** 2).mean()

def r2_masked_sample_loss(
        x: torch.Tensor,
        j: torch.Tensor,
        p: torch.Tensor,
        m: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor
) -> torch.Tensor:
    """A very unvectorized loss"""
    mask_inds = (j == 0).nonzero().squeeze()
    x_masked = torch.index_select(x, 0, mask_inds)
    a_masked = torch.index_select(a, 2, mask_inds)
    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    return ((x_masked - m_masked[0]) ** 2).sum()


nll_masked_batch_loss = inpainter_batch_loss_fn(nll_masked_sample_loss_v1)
r2_masked_batch_loss = inpainter_batch_loss_fn(r2_masked_sample_loss)