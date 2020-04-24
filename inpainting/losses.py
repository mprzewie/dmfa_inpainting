from typing import Callable, Tuple, Dict

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from time import time

from inpainting.datasets.mask_coding import UNKNOWN_LOSS

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

def nll_masked_batch_loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
        l_epsilon: float =1e-15,
):
    """A loss which allows for masks of varying size"""

    x_s, p_s, m_s, a_s, d_s, d_s_inv = zero_batch_at_mask_indices(X, J, P, M, A, D)
    

    x_minus_means = (x_s - m_s).unsqueeze(1)  # ?

    d_s_inv_rep = d_s_inv.unsqueeze(-2).repeat_interleave(dim=-2, repeats=a_s.shape[-2])
    a_s_t = a_s.transpose(1, 2)

    a_s_d_inv = a_s * d_s_inv_rep
    l_s = a_s_d_inv.bmm(a_s_t)
    l_s = l_s + torch.diag_embed(
        torch.ones_like(l_s[:, :, 0]) + l_epsilon
    )
    # equations (4) and (6) from https://papers.nips.cc/paper/7826-on-gans-and-gmms.pdf
    l_s_d = l_s.double()
    
    l_s_inv = l_s_d.inverse().float() 
    
    covs_inv_woodbury = torch.diag_embed(d_s_inv) - a_s_d_inv.transpose(1, 2).bmm(l_s_inv).bmm(a_s_d_inv)  
    # M.data(?)[:, range(100), range(100)] = d_inv (wektory, nie macierze diagonalne) - M[:, range(100), range(100)]

    log_dets_lemma = l_s_d.logdet().float() + (d_s + (d_s == 0)).log().sum(dim=1)
    # a hack: I add 1 where d_s == 0 so that d_s.log() is zero where d_s == 0
    log_noms = x_minus_means.bmm(covs_inv_woodbury).bmm(x_minus_means.transpose(1, 2)).reshape(-1)
    losses = p_s * (1 / 2) * (log_noms + log_dets_lemma + log_2pi * (d_s != 0).sum(dim=1))
    
#     print([t.sum().item() for t in [p_s, log_noms, log_dets_lemma, l_s.logdet(), l_s.log(), l_s.det()]])
#     print("----")
    comps = {
        "log_noms": log_noms, #.sum(),
        "log_dets_lemma": log_dets_lemma, #.sum(),
        "d": (d_s != 0), #.sum(),
        "l_s.logdet": l_s.logdet(),
        "l_s.det": l_s.det(),
    }
    
    if any([torch.isnan(t.sum()) for t in comps.values()]):
        print(comps)
        
        import matplotlib.pyplot as plt
        
        X_nan = X[torch.isnan(l_s.logdet())]
        
        fig, ax = plt.subplots(ncols=len(X_nan) + 1)
        for i, x in enumerate(X_nan):
            ax[i].imshow(x.cpu().numpy(), cmap="gray")
        plt.show()
        
    return losses.sum() / X.shape[0]


def zero_batch_at_mask_indices(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor
]:
    """
    Args:
        X: [b, c, h, w]
        J: [b, c, h, w]
        P: [b, mx]
        M: [b, mx, c*h*w]
        A: [b, mx, l, c*h*w]
        D: [b, mx, c*h*w]

    Returns:
        X: [b * mx, c*h*w]
        P: [b * mx]
        M: [b * mx, c*h*w]
        A: [b * mx, l, c*h*w]
        D: [b * mx, c*h*w]
        D_inv: [b * mx, c*h*w]
    """

    b = X.shape[0]
    chw = torch.tensor(X.shape[1:]).prod()
    mx = M.shape[1]
    X_b_chw = X.reshape(b, chw)
    J_b_chw = J.reshape(b, chw)
    l = A.shape[2]
    mask_inv = (J_b_chw == UNKNOWN_LOSS).float()
    X_zeroed = X_b_chw * mask_inv
    
    X_bmx_chw = X_zeroed.unsqueeze(1).repeat_interleave(mx, 1).reshape(b * mx, chw)
    A_zeroed = (A.transpose(0, 2) * mask_inv).transpose(0, 2)
    M_zeroed = (M.transpose(0, 1) * mask_inv).transpose(0, 1)
    D_trans = D.transpose(0, 1)
    D_inv_zeroed = (1 / D_trans * mask_inv).transpose(0, 1)
    D_zeroed = (D_trans * mask_inv).transpose(0, 1)
    A_bmx_l_chw = A_zeroed.reshape(b * mx, l, chw)
    M_bmx_chw = M_zeroed.reshape(b * mx, chw)
    D_bmx_chw = D_zeroed.reshape(b * mx, chw)
    D_inv_bmx_chw = D_inv_zeroed.reshape(b * mx, chw)
    P_bmx = P.reshape(b * mx)
    return X_bmx_chw, P_bmx, M_bmx_chw, A_bmx_l_chw, D_bmx_chw, D_inv_bmx_chw


def r2_total_batch_loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
) -> torch.Tensor:
    return ((X - M[:, 0, :]) ** 2).mean()


def r2_masked_batch_loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
) -> torch.Tensor:
    x_s, p_s, m_s, a_s, d_s = gather_batch_by_mask_indices(X, J, P, M, A, D)
    res = ((x_s - m_s)).pow(2).sum(dim=1)
    return (p_s * res).mean()


def nll_masked_batch_loss_same_size_masks(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
):
    """A loss which assumes that all masks are of the same size """
    #     t1 = time()

    x_s, p_s, m_s, a_s, d_s = gather_batch_by_mask_indices(X, J, P, M, A, D)

    x_minus_means = (x_s - m_s).unsqueeze(1)  # ?
    d_s_inv = 1 / d_s  # == najpierw  1 / d, a potem

    d_s_inv_rep = d_s_inv.unsqueeze(-2).repeat_interleave(dim=-2, repeats=a_s.shape[-2])
    a_s_t = a_s.transpose(1, 2)

    a_s_d_inv = a_s * d_s_inv_rep
    l_s = a_s_d_inv.bmm(a_s_t)
    l_s = l_s + torch.diag_embed(torch.ones_like(l_s[:, :, 0]))
    # equations (4) and (6) from https://papers.nips.cc/paper/7826-on-gans-and-gmms.pdf
    covs_inv_woodbury = torch.diag_embed(d_s_inv) - a_s_d_inv.transpose(1, 2).bmm(l_s.inverse()).bmm(
        a_s_d_inv)  # M.data(?)[:, range(100), range(100)] = d_inv (wektory, nie macierze diagonalne) - M[:, range(100), range(100)]
    log_dets_lemma = l_s.logdet() + d_s.log().sum(dim=1)
    log_noms = x_minus_means.bmm(covs_inv_woodbury).bmm(x_minus_means.transpose(1, 2)).reshape(-1)
    losses = p_s * (1 / 2) * (log_noms + log_dets_lemma + log_2pi * x_s.shape[1])

    return losses.sum() / X.shape[0]


def gather_batch_by_mask_indices(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Args:
        X: [b, c, h, w]
        J: [b, c, h, w]
        P: [b, mx]
        M: [b, mx, c*h*w]
        A: [b, mx, l, c*h*w]
        D: [b, mx, c*h*w]

    Returns:
        X: [b * mx, msk]
        P: [b * mx, msk]
        M: [b * mx, msk]
        A: [b * mx, l, msk]
        D: [b * mx, msk]
        msk - size of the mask
    """

    b = X.shape[0]
    chw = torch.tensor(X.shape[1:]).prod()
    mx = M.shape[1]
    X_b_chw = X.reshape(b, chw)
    J_b_chw = J.reshape(b, chw)
    l = A.shape[2]

    mask_inds_b, mask_inds_chw = (J_b_chw == UNKNOWN_LOSS).nonzero(as_tuple=True)
    msk = mask_inds_b.shape[0] // b
    X_b_msk = X_b_chw[mask_inds_b, mask_inds_chw].reshape(b, msk)
    X_bmx_msk = X_b_msk.unsqueeze(1).repeat_interleave(mx, 1).reshape(b * mx, msk)

    A_bmx_l_msk = A.transpose(
        1, 3
    )[mask_inds_b, mask_inds_chw].reshape(
        b, msk, l, mx).transpose(1, 3).reshape(b * mx, l, msk)

    M_bmx_msk = M.transpose(1, 2)[mask_inds_b, mask_inds_chw].reshape(b, msk, mx).transpose(1, 2).reshape(b * mx, msk)
    D_bmx_msk = D.transpose(1, 2)[mask_inds_b, mask_inds_chw].reshape(b, msk, mx).transpose(1, 2).reshape(b * mx, msk)

    P_bmx = P.reshape(b * mx)
    return X_bmx_msk, P_bmx, M_bmx_msk, A_bmx_l_msk, D_bmx_msk


def _nll_masked_sample_loss_v1(
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
    mask_inds = (j_hw == UNKNOWN_LOSS).nonzero().squeeze()
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


def _nll_masked_sample_loss_v2(
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
    l - covariance matrix width
    Args:
        x: [c, h, w]
        j: [c, h, w]
        p: [mx]
        m: [mx, c*h*w]
        a: [mx, l, c*h*w]
        d: [mx, c*h*w]

    Returns:

    """
    x_c_hw = x.reshape(-1)
    j_hw = j.reshape(-1)
    mask_inds = (j_hw == UNKNOWN_LOSS).nonzero().squeeze()
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


def _unvectorized_gather_sample(x: torch.Tensor,
                                j: torch.Tensor,
                                p: torch.Tensor,
                                m: torch.Tensor,
                                a: torch.Tensor,
                                d: torch.Tensor):
    x_c_hw = x.reshape(-1)
    j_hw = j.reshape(-1)
    mask_inds = (j_hw == UNKNOWN_LOSS).nonzero().squeeze()
    x_masked = torch.index_select(x_c_hw, 0, mask_inds).float()
    a_masked = torch.index_select(a, 2, mask_inds)
    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    return x_masked.unsqueeze(0).repeat([m_masked.shape[0], 1]), p, m_masked, a_masked, d_masked,


def _nll_masked_ubervectorized_batch_loss_v1(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
):
    """A loss which assumes that all masks are of the same size """
    x_s = []
    m_s = []
    d_s = []
    a_s = []
    p_s = []
    # TODO vectorize index selection
    for (x, j, p, m, a, d) in zip(X, J, P, M, A, D):
        x_p, p_p, m_p, a_p, d_p, = _unvectorized_gather_sample(x, j, p, m, a, d)
        x_s.extend(x_p)
        m_s.extend(m_p)
        d_s.extend(d_p)
        a_s.extend(a_p)
        p_s.extend(p_p)

    x_s, m_s, d_s, a_s, p_s = [torch.stack(t) for t in [x_s, m_s, d_s, a_s, p_s]]

    x_minus_means = (x_s - m_s).unsqueeze(1)  # ?
    d_s_inv = torch.diag_embed(d_s).inverse()  # == najpierw  1 / d, a potem
    l_s = a_s.bmm(d_s_inv).bmm(a_s.transpose(1, 2))  # zamiast macierzy, a * D -1 (elemnt-wise)
    l_s = l_s + torch.diag_embed(torch.ones_like(l_s[:, :, 0]))
    # equations (4) and (6) from https://papers.nips.cc/paper/7826-on-gans-and-gmms.pdf
    covs_inv_woodbury = d_s_inv - d_s_inv.bmm(a_s.transpose(1, 2)).bmm(l_s.inverse()).bmm(a_s).bmm(
        d_s_inv)  # M.data(?)[:, range(100), range(100)] = d_inv (wektory, nie macierze diagonalne) - M[:, range(100), range(100)]
    log_dets_lemma = l_s.det().log() + (d_s).log().sum(dim=1)  # .log_det()
    log_noms = x_minus_means.bmm(covs_inv_woodbury).bmm(x_minus_means.transpose(1, 2)).reshape(-1)
    losses = p_s * (1 / 2) * (log_noms + log_dets_lemma + log_2pi * x_s.shape[1])

    return losses.sum() / X.shape[0]


def _batch_loss_fn(
        sample_loss: Callable[[
                                  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
                              ], torch.Tensor] = _nll_masked_sample_loss_v1) -> InpainterLossFn:
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


def _r2_masked_sample_loss(
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
    mask_inds = (j_hw == UNKNOWN_LOSS).nonzero().squeeze()
    x_masked = torch.index_select(x_c_hw, 1, mask_inds)

    m_masked, d_masked = [
        torch.index_select(t, 1, mask_inds)
        for t in [m, d]
    ]
    return torch.stack([p_i * ((x_masked - m_i) ** 2).sum() for (p_i, m_i) in zip(p, m_masked)]).sum()


_nll_masked_batch_loss = _batch_loss_fn(_nll_masked_sample_loss_v1)
_r2_masked_batch_loss = _batch_loss_fn(_r2_masked_sample_loss)

def _nll_masked_sample_loss_v0(
        x: torch.Tensor,
        j: torch.Tensor,
        p: torch.Tensor,
        m: torch.Tensor,
        a: torch.Tensor,
        d: torch.Tensor
) -> torch.Tensor:
    x_c_hw = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
    j_hw = j.reshape(-1)
    mask_inds = (j_hw == UNKNOWN_LOSS).nonzero().squeeze()
    x_masked = torch.index_select(x_c_hw, 1, mask_inds)
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

def nll_masked_batch_loss_components(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """A loss which allows for masks of varying size"""

    x_s, p_s, m_s, a_s, d_s, d_s_inv = zero_batch_at_mask_indices(X, J, P, M, A, D)

    x_minus_means = (x_s - m_s).unsqueeze(1)  # ?

    d_s_inv_rep = d_s_inv.unsqueeze(-2).repeat_interleave(dim=-2, repeats=a_s.shape[-2])
    a_s_t = a_s.transpose(1, 2)

    a_s_d_inv = a_s * d_s_inv_rep
    l_s = a_s_d_inv.bmm(a_s_t)
    l_s = l_s + torch.diag_embed(torch.ones_like(l_s[:, :, 0]))
    # equations (4) and (6) from https://papers.nips.cc/paper/7826-on-gans-and-gmms.pdf
    l_s_inv = l_s.inverse()
    
    covs_inv_woodbury = torch.diag_embed(d_s_inv) - a_s_d_inv.transpose(1, 2).bmm(l_s_inv).bmm(a_s_d_inv)  
    log_dets_lemma = l_s.logdet() + (d_s + (d_s == 0)).log().sum(dim=1)
    log_noms = x_minus_means.bmm(covs_inv_woodbury).bmm(x_minus_means.transpose(1, 2)).reshape(-1)
    log_2_pi_res = log_2pi * (d_s != 0).sum(dim=1)
    result= {
        comp_name: (p_s * (1 / 2) * comp).sum() / X.shape[0]
        for comp_name, comp in [
            ("log_noms", log_noms),
            ("x_minus_means", x_minus_means.sum(dim=[1,2])),
            ("log_dets", log_dets_lemma),
            ("log_2_pi", log_2_pi_res ),
            ("x_minus_means_2", (x_minus_means ** 2).sum(dim=[1,2])),
        ]
    }

    return result

def loss_with_mixup(loss_fn, mixup_rate = 1/ 2):
    def loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
    ):
        xs = X.shape[0]
        n = int(xs * mixup_rate)
        perm = torch.randperm(n)
        X_perm = X.clone()
        X_perm[-n:] = X_perm[-n:][perm]
        return loss_fn(X_perm,J,P,M,A,D)
    return loss