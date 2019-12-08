from typing import Callable

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from time import time
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
    """
    c - channels
    h - height
    w - width
    mx - n_mixes
    aw - covariance matrix width
    Args:
        x: [c, h, w]
        j: [h, w]
        p: [mx]
        m: [mx, c*h*w]
        a: [mx, aw, c*h*w]
        d: [mx, c*h*w]

    Returns:

    """
    x_c_hw = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    j_hw = j.reshape(-1)
    mask_inds = (j_hw == 0).nonzero().squeeze()
    x_masked = torch.index_select(x_c_hw, 1, mask_inds)
    a_masked = torch.index_select(a, 2, mask_inds)
    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    covs = torch.bmm(a_masked.transpose(1, 2), a_masked)
    losses_for_p = []
    for (p_i, m_i, d_i, cov_i) in zip(p, m_masked, d_masked, covs):
        if cov_i.shape[1] > 0:
            cov = cov_i + torch.diag(d_i)
            mvn_d = MultivariateNormal(m_i, cov)  # calculate this manually
            loss_for_p = - mvn_d.log_prob(x_masked.float())
            losses_for_p.append(p_i * loss_for_p)
    return torch.stack(losses_for_p).sum()


log_2pi = torch.log(torch.tensor(2 * np.pi))


def nll_masked_sample_loss_v2(
        x: torch.Tensor,
        j: torch.Tensor,
        p: torch.Tensor,
        m: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor
) -> torch.Tensor:
    """
    A potentially vectorized version of v1
    c - channels
    h - height
    w - width
    mx - n_mixes
    aw - covariance matrix width
    Args:
        x: [c, h, w]
        j: [c, h, w]
        p: [mx]
        m: [mx, c*h*w]
        a: [mx, aw, c*h*w]
        d: [mx, c*h*w]

    Returns:

    """
    x_c_hw = x.reshape(-1)
    j_hw = j.reshape(-1)
    mask_inds = (j_hw == 0).nonzero().squeeze()
    x_masked = torch.index_select(x_c_hw, 0, mask_inds).float()
    a_masked = torch.index_select(a, 2, mask_inds)
    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    covs = a_masked.transpose(1, 2).bmm(a_masked) + torch.diag_embed(d_masked)
    x_minus_means = (x_masked - m_masked).unsqueeze(1)
    log_noms = x_minus_means.bmm(covs.inverse()).bmm(x_minus_means.transpose(1, 2))
    log_dets = covs.det().log()
    losses = p * (1 / 2) * (log_noms + log_dets + log_2pi * x_masked.shape[0])
    return losses.sum()


def preprocess_sample(x: torch.Tensor,
                      j: torch.Tensor,
                      p: torch.Tensor,
                      m: torch.Tensor,
                      a: torch.Tensor,
                      d: torch.Tensor):
    x_c_hw = x.reshape(-1)
    j_hw = j.reshape(-1)
    mask_inds = (j_hw == 0).nonzero().squeeze()
    x_masked = torch.index_select(x_c_hw, 0, mask_inds).float()
    a_masked = torch.index_select(a, 2, mask_inds)
    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    return x_masked.unsqueeze(0).repeat([m_masked.shape[0], 1]), p, m_masked, a_masked, d_masked,


def nll_masked_ubervectorized_batch_loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
):
    """A loss which assumes that all masks are of the same size """
    t1 = time()
    x_s = []
    m_s = []
    d_s = []
    a_s = []
    p_s = []
    # TODO vectorize index selection
    for (x, j, p, m, a, d) in zip(X, J, P, M, A, D):
        x_p, p_p, m_p, a_p, d_p, = preprocess_sample(x, j, p, m, a, d)
        x_s.extend(x_p)
        m_s.extend(m_p)
        d_s.extend(d_p)
        a_s.extend(a_p)
        p_s.extend(p_p)
    
    x_s, m_s, d_s, a_s, p_s = [torch.stack(t) for t in [x_s, m_s, d_s, a_s, p_s]]
    
    t2 = time()
    
    x_minus_means = (x_s - m_s).unsqueeze(1) # ?
    d_s_inv = torch.diag_embed(d_s).inverse() # == najpierw  1 / d, a potem
    l_s = a_s.bmm(d_s_inv).bmm(a_s.transpose(1,2)) # zamiast macierzy, a * D -1 (elemnt-wise)
    l_s = l_s + torch.diag_embed(torch.ones_like(l_s[:, :, 0]))
    # equations (4) and (6) from https://papers.nips.cc/paper/7826-on-gans-and-gmms.pdf
    covs_inv_woodbury = d_s_inv - d_s_inv.bmm(a_s.transpose(1,2)).bmm(l_s.inverse()).bmm(a_s).bmm(d_s_inv) # M.data(?)[:, range(100), range(100)] = d_inv (wektory, nie macierze diagonalne) - M[:, range(100), range(100)]
    log_dets_lemma = l_s.det().log() + (d_s).log().sum(dim=1) # .log_det()
    log_noms = x_minus_means.bmm(covs_inv_woodbury).bmm(x_minus_means.transpose(1, 2)).reshape(-1)
    losses = p_s * (1 / 2) * (log_noms + log_dets_lemma + log_2pi * x_s.shape[1])
    
    t3 = time()
    print("v1")
    print("mask indices selection", t2 - t1)
    print("actual NLL calculation", t3 - t2)
    return losses.sum() / X.shape[0]

def nll_masked_ubervectorized_batch_loss_v2(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
):
    """A loss which assumes that all masks are of the same size """
    t1 = time()
    x_s = []
    m_s = []
    d_s = []
    a_s = []
    p_s = []
    # TODO vectorize index selection
    for (x, j, p, m, a, d) in zip(X, J, P, M, A, D):
        x_p, p_p, m_p, a_p, d_p, = preprocess_sample(x, j, p, m, a, d)
        x_s.extend(x_p)
        m_s.extend(m_p)
        d_s.extend(d_p)
        a_s.extend(a_p)
        p_s.extend(p_p)
    
    x_s, m_s, d_s, a_s, p_s = [torch.stack(t) for t in [x_s, m_s, d_s, a_s, p_s]]
    
    t2 = time()
    
    x_minus_means = (x_s - m_s).unsqueeze(1) # ?
    d_s_inv = 1 / d_s # == najpierw  1 / d, a potem
    
    d_s_inv_rep = d_s_inv.unsqueeze(-2).repeat_interleave(dim=-2, repeats=a_s.shape[-2])
    d_s_rep = d_s.unsqueeze(-2).repeat_interleave(dim=-2, repeats=a_s.shape[-2])
    a_s_t = a_s.transpose(1,2)
    
    a_s_d_inv = a_s * d_s_inv_rep
    l_s = a_s_d_inv.bmm(a_s_t) # zamiast macierzy, a * D -1 (elemnt-wise)
    l_s = l_s + torch.diag_embed(torch.ones_like(l_s[:, :, 0]))
    # equations (4) and (6) from https://papers.nips.cc/paper/7826-on-gans-and-gmms.pdf
    covs_inv_woodbury = torch.diag_embed(d_s_inv) - a_s_d_inv.transpose(1,2).bmm(l_s.inverse()).bmm(a_s_d_inv) # M.data(?)[:, range(100), range(100)] = d_inv (wektory, nie macierze diagonalne) - M[:, range(100), range(100)]
    log_dets_lemma = l_s.logdet() + (d_s).log().sum(dim=1) 
    log_noms = x_minus_means.bmm(covs_inv_woodbury).bmm(x_minus_means.transpose(1, 2)).reshape(-1)
    losses = p_s * (1 / 2) * (log_noms + log_dets_lemma + log_2pi * x_s.shape[1])
    
    t3 = time()
    print("v2")
    print("mask indices selection", t2 - t1)
    print("actual NLL calculation", t3 - t2)
    return losses.sum() / X.shape[0]

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
    x_c_hw = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    j_hw = j.reshape(-1)
    mask_inds = (j_hw == 0).nonzero().squeeze()
    x_masked = torch.index_select(x_c_hw, 1, mask_inds)

    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    return torch.stack([p_i * ((x_masked - m_i) ** 2).sum() for (p_i, m_i) in zip(p, m_masked)]).sum()


nll_masked_batch_loss = inpainter_batch_loss_fn(nll_masked_sample_loss_v2)
r2_masked_batch_loss = inpainter_batch_loss_fn(r2_masked_sample_loss)
