from collections import defaultdict
from typing import Optional, List, Dict, Callable

import numpy as np

from inpainting.datasets.mask_coding import KNOWN, UNKNOWN_LOSS
from inpainting.utils import inpainted
from inpainting.visualizations.digits import digit_with_mask as vis_digit_mask
import matplotlib.pyplot as plt


def row_length(
        x: np.ndarray, j: np.ndarray, p: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray, y: int,
) -> int:
    """
    Args:
        x: [c, h, w]
        j: [c, h, w]
        p: [mx]
        m: [mx, c*h*w]
        a: [mx, l, c*h*w]
        d: [mx, c*h*w]
    """
    mx, l = a.shape[:2]
    return sum([
        4,  # x, j, x_input, x_inpainted
        mx,  # m
        mx * l,  # a
        mx  # d
    ])


def visualize_sample(
        x: np.ndarray, j: np.ndarray, p: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray, y: int,
        title_prefixes: Dict[int, str], ax_row: Optional[np.ndarray] = None
):
    """
    Args:
        x: [c, h, w]
        j: [c, h, w]
        p: [mx]
        m: [mx, c*h*w]
        a: [mx, l, c*h*w]
        d: [mx, c*h*w]
        title_prefix:
        ax_row:

    Returns:

    """
    row_len = row_length(x, j, p, m, a, d, y)

    if ax_row is None:
        fig, ax_row = plt.subplots(1, row_len)

    assert ax_row.shape[0] >= row_len
    if title_prefixes is None:
        title_prefixes = dict()

    title_prefixes = defaultdict(str, title_prefixes)
    ax_x_original = ax_row[0]

    img_shape = (x.shape[1], x.shape[2])
    ax_x_original.imshow(
        x.reshape(*img_shape),
        cmap="gray")
    ax_x_original.set_title(f"{title_prefixes[0]}y_gt = {y}")

    ax_x_masked = ax_row[1]
    vis_digit_mask(x, j, ax_x_masked)
    ax_x_masked.set_title(title_prefixes[1])

    ax_x_input = ax_row[2]
    ax_x_input.imshow(
        (x * (j==KNOWN)).reshape(*img_shape), cmap="gray",  vmin=0, vmax=1
    )
    ax_x_input.set_title("input to model")


    ax_inpainted = ax_row[3]
    m_ind = np.random.choice(np.arange(m.shape[0]), p=p)
    m_inp = m[m_ind].reshape(x.shape)
    x_inp = inpainted(x, j, m_inp)
    j_inp = j.copy()
    j_inp[j_inp==UNKNOWN_LOSS] = KNOWN
    vis_digit_mask(x_inp, j_inp, ax_inpainted)
    # ax_inpainted.imshow(x_inp.reshape(*img_shape), cmap="gray", vmin=0, vmax=1)
    ax_inpainted.set_title("j==unk inpainted")

    for i, m_ in enumerate(m):
        ax_m = ax_row[4 + i]
        ax_m.imshow(m_.reshape(*img_shape), cmap="gray", vmin=0, vmax=1)
        p_form = int(p[i] * 100) / 100
        chosen = "cho " if i == m_ind else ""
        ax_m.set_title(f"{chosen}M_{i}, p={p_form}")

    for i, a_ in enumerate(a):
        for j, a_l in enumerate(a_):
            offset = 4 + m.shape[0] + a.shape[1] * i + j
            ax_a_l = ax_row[offset]
            ax_a_l.imshow(a_l.reshape(*img_shape), cmap="gray")
            ttl = f"a_{i}_{j} "
            ax_a_l.set_title(ttl + "m = {0:.2f}".format(np.mean(a_l)))

    for i, d_ in enumerate(d):
        offset = 4 + m.shape[0] + a.shape[0] * a.shape[1] + i
        ax_d = ax_row[offset]
        ax_d.imshow(d_.reshape(*img_shape), cmap="gray")
        ttl = f"d_{i} "
        ax_d.set_title(ttl + "m = {0:.2f}".format(np.mean(d_)))
    for ax in ax_row:
        ax.axis("off")


def cov_sample_no_d(
        x: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray
) -> np.ndarray:
    return np.random.multivariate_normal(
        m, a.T @ a
    ).reshape(x.shape)


def gans_gmms_sample_no_d(
        x: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """Sampling like https://github.com/eitanrich/gans-n-gmms/blob/master/utils/mfa.py#L64"""
    return (
            np.random.normal(size=a.shape[0]) @ a + m
    ).reshape(x.shape)


def visualize_distribution_samples(
        x: np.ndarray, j: np.ndarray, p: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray, y: int,
        ax_row: Optional[np.ndarray] = None,
        sample_fn: Callable[[
                                np.ndarray,
                                np.ndarray,
                                np.ndarray,
                                np.ndarray
                            ], np.ndarray] = gans_gmms_sample_no_d
):
    """
    Args:
        x: [c, h, w]
        j: [c, h, w]
        p: [mx]
        m: [mx, c*h*w]
        a: [mx, l, c*h*w]
        d: [mx, c*h*w]
        title_prefix:
        ax_row:
        sample_fn:

    Returns:

    """

    row_len = 3 + 3 * m.shape[0]
    if ax_row is None:
        fig, ax_row = plt.subplots(1, row_len)

    assert ax_row.shape[0] >= row_len

    ax_x_original = ax_row[0]

    img_shape = (x.shape[1], x.shape[2])
    ax_x_original.imshow(
        x.reshape(*img_shape),
        cmap="gray")

    ax_x_masked = ax_row[1]
    vis_digit_mask(x, j, ax_x_masked)

    ax_x_input = ax_row[2]
    ax_x_input.imshow(
        (x * (j == KNOWN)).reshape(*img_shape), cmap="gray", vmin=0, vmax=1
    )
    ax_x_input.set_title("input to model")

    for i, (m_, a_, d_) in enumerate(zip(m, a, d)):
        sampled_fill = sample_fn(x, m_, a_, d_)

        ax_m = ax_row[3 + 3 * i]
        ax_m.imshow(m_.reshape(*img_shape), cmap="gray", vmin=0, vmax=1)
        ax_m.set_title(f"m_{i}")

        ax_fill = ax_row[3 + 3 * i + 1]
        ax_fill.imshow(sampled_fill.reshape(*img_shape), cmap="gray", vmin=0, vmax=1)
        ax_fill.set_title(f"sampled_{i}")

        ax_inp = ax_row[3 + 3 * i + 2]
        x_inp = inpainted(x, j, sampled_fill)
        j_inp = j.copy()
        j_inp[j_inp == UNKNOWN_LOSS] = KNOWN
        vis_digit_mask(x_inp, j_inp, ax_inp)
        ax_inp.set_title(f"inpainted_{i}")

    for ax in ax_row:
        ax.axis("off")
