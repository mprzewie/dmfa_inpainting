from collections import defaultdict
from typing import Optional, List, Dict

import numpy as np

from inpainting.visualizations.digits import digit_with_mask as vis_digit_mask


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
        3, # x, j, x_inp
        mx, # m
        # mx * l, # a
        # mx # d
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
    assert ax_row.shape[0] >= row_length(x, j, p, m, a, d, y)
    if ax_row is None:
        raise TypeError()
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

    ax_inpainted = ax_row[2]
    x_inp = x.copy()
    m_ind = np.random.choice(np.arange(m.shape[0]), p=p)
    m_inp = m[m_ind].reshape(x.shape)
    x_inp[j == 0] = m_inp[j == 0]
    ax_inpainted.imshow(x_inp.reshape(*img_shape), cmap="gray", vmin=0, vmax=1)
    ax_inpainted.set_title(title_prefixes[2])

    for i, m_ in enumerate(m):
        ax_m = ax_row[3 + i]
        ax_m.imshow(m_.reshape(*img_shape), cmap="gray", vmin=0, vmax=1)
        p_form = int(p[i] * 100) / 100
        chosen = "chosen " if i == m_ind else ""
        ax_m.set_title(f"{chosen}M_{i}, p={p_form}")

    for ax in ax_row:
        ax.axis("off")