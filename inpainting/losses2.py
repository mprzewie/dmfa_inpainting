"""This module contains more optimized NLL implementation than losses.py"""
from collections import Callable
from typing import Tuple

import torch

from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS
from inpainting.losses import (
    InpainterLossFn,
    zero_batch_at_mask_indices,
    log_2pi,
    gather_batch_by_mask_indices,
)


def nll_calc_woodbury(x_s, p_s, m_s, a_s, d_s, return_sum: bool = True):
    """returns NLL for each element in batch!"""
    #     x_s, p_s, m_s, a_s, d_s = [t.double() for t in [x_s, p_s, m_s, a_s, d_s]]
    d_s_inv = (1 / (d_s + (d_s == 0))) * (d_s != 0)
    x_minus_means = (x_s - m_s).unsqueeze(1)
    d_s_inv_rep = d_s_inv.unsqueeze(-2).repeat_interleave(dim=-2, repeats=a_s.shape[-2])
    a_s_t = a_s.transpose(1, 2)

    a_s_d_inv = a_s * d_s_inv_rep
    l_s = a_s_d_inv.bmm(a_s_t)
    l_s = l_s + torch.diag_embed(torch.ones_like(l_s[:, :, 0]))
    # equations (4) and (6) from https://papers.nips.cc/paper/7826-on-gans-and-gmms.pdf
    covs_inv_woodbury = torch.diag_embed(d_s_inv) - a_s_d_inv.transpose(1, 2).bmm(
        l_s.inverse()
    ).bmm(a_s_d_inv)

    log_dets_lemma = l_s.logdet() + (d_s + (d_s == 0)).log().sum(
        dim=1
    )  # a hack: I add 1 where d_s == 0 so that d_s.log() is zero where d_s == 0
    log_noms = (
        x_minus_means.bmm(covs_inv_woodbury)
        .bmm(x_minus_means.transpose(1, 2))
        .reshape(-1)
    )
    losses = (0.5 * (log_noms + log_dets_lemma + log_2pi * (d_s != 0).sum(dim=1))) + p_s

    if return_sum:
        losses = losses.sum()
    return losses.float()


def mse(x_s, p_s, m_s, a_s, d_s):
    return ((x_s - m_s) ** 2).sum()


def signed_difference_mean(x_s, p_s, m_s, a_s, d_s):
    return (x_s - m_s).mean()


def signed_difference_std(x_s, p_s, m_s, a_s, d_s):
    return (x_s - m_s).std()


def buffered_gather_batch_by_mask_indices(
    X: torch.Tensor,
    J: torch.Tensor,
    P: torch.Tensor,
    M: torch.Tensor,
    A: torch.Tensor,
    D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        P: [b * mx]
        M: [b * mx, msk]
        A: [b * mx, l, msk]
        D: [b * mx, msk]

        where: msk - size of the mask
    """
    b = X.shape[0]
    chw = torch.tensor(X.shape[1:]).prod()
    mx = M.shape[1]
    X_b_chw = X.reshape(b, chw)
    J_b_chw = J.reshape(b, chw)
    l = A.shape[2]
    J_b_chw_msk = J_b_chw == UNKNOWN_LOSS
    mask_sizes = J_b_chw_msk.sum(dim=1)
    msk = mask_sizes.max().item()
    fake_masks_sizes = msk - mask_sizes
    non_masks_inds_b, non_masks_inds_chw = (J_b_chw_msk == 0).nonzero(as_tuple=True)

    fillups = [
        torch.stack(
            [
                non_masks_inds_b[non_masks_inds_b == i][:fms],
                non_masks_inds_chw[non_masks_inds_b == i][:fms],
            ]
        ).T
        for i, fms in enumerate(fake_masks_sizes)
    ]

    fillups = torch.cat(fillups)
    J_b_chw_msk[fillups[:, 0], fillups[:, 1]] = UNKNOWN_NO_LOSS

    mask_inds_b, mask_inds_chw = (J_b_chw_msk).nonzero(as_tuple=True)

    J_b_msk = J_b_chw[mask_inds_b, mask_inds_chw] == UNKNOWN_LOSS

    X_b_msk = X_b_chw[mask_inds_b, mask_inds_chw].reshape(b, msk)
    X_bmx_msk = X_b_msk.unsqueeze(1).repeat_interleave(mx, 1).reshape(b * mx, msk)

    A_bmx_l_msk = (
        (
            (
                A.transpose(1, 3)[mask_inds_b, mask_inds_chw].transpose(0, 2) * J_b_msk
            ).transpose(0, 2)
        )
        .reshape(b, msk, l, mx)
        .transpose(1, 3)
        .reshape(b * mx, l, msk)
    )

    M_bmx_msk = (
        (
            (
                M.transpose(1, 2)[mask_inds_b, mask_inds_chw].transpose(0, 1) * J_b_msk
            ).transpose(0, 1)
        )
        .reshape(b, msk, mx)
        .transpose(1, 2)
        .reshape(b * mx, msk)
    )
    D_bmx_msk = (
        (
            (
                D.transpose(1, 2)[mask_inds_b, mask_inds_chw].transpose(0, 1) * J_b_msk
            ).transpose(0, 1)
        )
        .reshape(b, msk, mx)
        .transpose(1, 2)
        .reshape(b * mx, msk)
    )
    P_bmx = P.reshape(b * mx)

    return X_bmx_msk, P_bmx, M_bmx_msk, A_bmx_l_msk, D_bmx_msk


def loss_factory(
    gathering_fn: Callable = zero_batch_at_mask_indices,
    calc_fn: Callable = nll_calc_woodbury,
) -> InpainterLossFn:
    def loss(
        X: torch.Tensor,
        J: torch.Tensor,
        P: torch.Tensor,
        M: torch.Tensor,
        A: torch.Tensor,
        D: torch.Tensor,
    ):
        x_s, p_s, m_s, a_s, d_s = gathering_fn(X, J, P, M, A, D)
        result = calc_fn(x_s, p_s, m_s, a_s, d_s) / X.shape[0]

        return result

    return loss


nll_zero = loss_factory(zero_batch_at_mask_indices, nll_calc_woodbury)
nll_gather = loss_factory(gather_batch_by_mask_indices, nll_calc_woodbury)
nll_buffered = loss_factory(gathering_fn=buffered_gather_batch_by_mask_indices)

mse_buffered = loss_factory(
    gathering_fn=buffered_gather_batch_by_mask_indices, calc_fn=mse
)
signed_difference_mean_buffered = loss_factory(
    gathering_fn=buffered_gather_batch_by_mask_indices, calc_fn=signed_difference_mean
)
signed_difference_std_buffered = loss_factory(
    gathering_fn=buffered_gather_batch_by_mask_indices, calc_fn=signed_difference_std
)


def nll_plus_mse_calc(nll_weight=1, mse_weight=0) -> Callable:
    def calc_fn(x_s, p_s, m_s, a_s, d_s):
        nll_val = nll_calc_woodbury(x_s, p_s, m_s, a_s, d_s)
        mse_val = mse(x_s, p_s, m_s, a_s, d_s)

        return (nll_weight * nll_val) + (mse_weight * mse_val)

    return calc_fn


def nll_plus_mse_weighted_loss(nll_weight=1, mse_weight=0) -> Callable:
    return loss_factory(
        gathering_fn=buffered_gather_batch_by_mask_indices,
        # gathering_fn=gather_batch_by_mask_indices,
        calc_fn=nll_plus_mse_calc(nll_weight, mse_weight),
    )
