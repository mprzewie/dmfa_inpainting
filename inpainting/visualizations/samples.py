from collections import defaultdict
from typing import Optional, List, Dict, Callable

import numpy as np

from inpainting.datasets.mask_coding import KNOWN, UNKNOWN_LOSS
from inpainting.utils import inpainted
from inpainting.visualizations.digits import img_with_mask as vis_digit_mask
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm
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
        3,  # x, j, x_input
        4*mx, #m, x_inp_m, s, x_inp_s
        
        mx * l,  # a
        mx  # d,
       
    ])

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


def visualize_sample(
        x: np.ndarray, j: np.ndarray, p: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray, y: int,
        title_prefixes: Dict[int, str] = None, ax_row: Optional[np.ndarray] = None,
        drawing_fn: Callable[[np.ndarray, np.ndarray, plt.Axes], None] = vis_digit_mask,
        sample_fn=gans_gmms_sample_no_d
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


    drawing_fn(x, ax=ax_x_original)
    ax_x_original.set_title(f"{title_prefixes[0]} original")

    ax_x_masked = ax_row[1]
    drawing_fn(x, j, ax_x_masked)
    ax_x_masked.set_title(title_prefixes[1])

    ax_x_input = ax_row[2]
    drawing_fn(
        (x * (j == KNOWN)),
        ax=ax_x_input
    )

    ax_x_input.set_title("input to model")

    for i, m_ in enumerate(m):
        ax_m = ax_row[3 + 4*i]
        drawing_fn(
            m_.reshape(*x.shape),
            ax=ax_m
        )
        p_form = int(p[i] * 100) / 100
        ax_m.set_title(f"m_{i}, p={p_form}")
        
        ax_x_inp_m = ax_row[3 + 4*i+1]
        m_inp = m_.reshape(x.shape)
        x_inp = inpainted(x, j, m_inp)
        j_inp = j.copy()
        j_inp[j_inp==UNKNOWN_LOSS] = KNOWN
        drawing_fn(
            x_inp,
            j_inp,
            ax_x_inp_m
        )
        ax_x_inp_m.set_title(f"x_inp_m_{i}")
        a_ = a[i]
        d_ = d[i]
        
        s = sample_fn(x, m_, a_, d_)
        ax_s= ax_row[3 + 4*i+2]
        
        drawing_fn(
            s,
            ax=ax_s
        )
        ax_s.set_title(f"s_{i}")
        
        x_inp_s = inpainted(x, j, s)
        j_inp = j.copy()
        j_inp[j_inp == UNKNOWN_LOSS] = KNOWN
        
        ax_x_inp_s = ax_row[3 + 4*i+3]
        drawing_fn(
            x_inp_s,
            j=j_inp,
            ax=ax_x_inp_s
        )
        ax_x_inp_s.set_title(f"x_inp_s{i}")

    for i, a_ in enumerate(a):
        for j, a_l in enumerate(a_):
            offset = 3 + 4*m.shape[0] + a.shape[1] * i + j
            ax_a_l = ax_row[offset]
            drawing_fn(
                a_l.reshape(*x.shape) + (1/2),
                ax=ax_a_l,
                clip=False
            )
            ttl = f"a_{i}_{j} "
            ax_a_l.set_title(ttl + "m = {0:.2f}".format(np.mean(a_l)))

    for i, d_ in enumerate(d):
        offset = 3 + 4*m.shape[0] + a.shape[0] * a.shape[1] + i
        ax_d = ax_row[offset]
        drawing_fn(
            d_.reshape(*x.shape),
            ax=ax_d
        )
        ttl = f"d_{i} "
        ax_d.set_title(ttl + "m = {0:.2f}".format(np.mean(d_)))
    for ax in ax_row:
        ax.axis("off")





def visualize_distribution_samples(
        x: np.ndarray, j: np.ndarray, p: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray, y: int,
        ax_row: Optional[np.ndarray] = None,
        sample_fn: Callable[[
                                np.ndarray,
                                np.ndarray,
                                np.ndarray,
                                np.ndarray
                            ], np.ndarray] = gans_gmms_sample_no_d,
        drawing_fn: Callable[[np.ndarray, np.ndarray, plt.Axes], None] = vis_digit_mask

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

    row_len = 3 + 4 * m.shape[0]
    if ax_row is None:
        fig, ax_row = plt.subplots(1, row_len)

    assert ax_row.shape[0] >= row_len

    ax_x_original = ax_row[0]

    drawing_fn(x, ax=ax_x_original)

    ax_x_masked = ax_row[1]
    drawing_fn(x, j, ax_x_masked)

    ax_x_input = ax_row[2]
    drawing_fn(
        (x * (j == KNOWN)),
        ax=ax_x_input
    )
    ax_x_input.set_title("input to model")

    for i, (m_, a_, d_) in enumerate(zip(m, a, d)):
        sampled_fill = sample_fn(x, m_, a_, d_)

        ax_m = ax_row[3 + 3 * i]
        m_ = m_.reshape(*x.shape)
        drawing_fn(
            m_,
            ax=ax_m
        )
        ax_m.set_title(f"m_{i}")

        ax_m_inp = ax_row[3 + 3* i + 1]
        x_inp = inpainted(x, j, m_)
        j_inp = j.copy()
        j_inp[j_inp == UNKNOWN_LOSS] = KNOWN
        drawing_fn(
            x_inp,
            j_inp,
            ax_m_inp
        )
        ax_m_inp.set_title(f"inp_m_{i}")

        ax_fill = ax_row[3 + 3 * i + 2]
        drawing_fn(
            sampled_fill.reshape(*x.shape),
            ax=ax_fill
        )
        ax_fill.set_title(f"s_{i}")

        ax_inp = ax_row[3 + 3 * i + 3]
        x_inp = inpainted(x, j, sampled_fill)
        j_inp = j.copy()
        j_inp[j_inp == UNKNOWN_LOSS] = KNOWN
        drawing_fn(
            x_inp,
            j_inp,
            ax_inp
        )
        ax_inp.set_title(f"x_inp_s_{i}")

    for ax in ax_row:
        ax.axis("off")

        

def visualize_sample_for_paper(
        our_results: Tuple,
        torch_mfa_results: Tuple,
        ax_row: Optional[np.ndarray] = None,
        drawing_fn: Callable[[np.ndarray, np.ndarray, plt.Axes], None] = vis_digit_mask,
        sample_fn=gans_gmms_sample_no_d
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
    x, j, p, m, a, d, y = our_results
    row_len = row_length(x, j, p, m, a, d, y)
    if ax_row is None:
        fig, ax_row = plt.subplots(1, row_len)

#     if title_prefixes is None:
#         title_prefixes = dict()

#     title_prefixes = defaultdict(str, title_prefixes)
    ax_x_original = ax_row[0]


    drawing_fn(x, ax=ax_x_original)

    ax_x_masked = ax_row[1]
    drawing_fn(x, j, ax_x_masked)
    
    if torch_mfa_results["inpainted_means_0"].shape[2] == 1:
        ax_row[2].imshow(torch_mfa_results["inpainted_means_0"].squeeze(), cmap="gray", vmin=0, vmax=1)
    else:
         ax_row[2].imshow(torch_mfa_results["inpainted_means_0"], vmin=0, vmax=1)
#     drawing_fn(, None, ax_row[2])
#     ax_x_masked.set_title(title_prefixes[1])

#     ax_x_input = ax_row[1]
#     drawing_fn(
#         (x * (j == KNOWN)),
#         ax=ax_x_input
#     )

#     ax_x_input.set_title("input to model")

    for i, m_ in enumerate(m):
        ax_m = ax_row[3 + 2*i+1]
        drawing_fn(
            m_.reshape(*x.shape),
            ax=ax_m
        )
        p_form = int(p[i] * 100) / 100
#         ax_m.set_title(f"m_{i}, p={p_form}")
        
        ax_x_inp_m = ax_row[3 + 2*i]
        m_inp = m_.reshape(x.shape)
        x_inp = inpainted(x, j, m_inp)
        j_inp = j.copy()
        j_inp[j_inp==UNKNOWN_LOSS] = KNOWN
        drawing_fn(
            x_inp,
            j_inp,
            ax_x_inp_m
        )
#         ax_x_inp_m.set_title(f"x_inp_m_{i}")
#         a_ = a[i]
#         d_ = d[i]
        
#         s = sample_fn(x, m_, a_, d_)
#         ax_s= ax_row[3 + 4*i+2]
        
#         drawing_fn(
#             s,
#             ax=ax_s
#         )
#         ax_s.set_title(f"s_{i}")
        
#         x_inp_s = inpainted(x, j, s)
#         j_inp = j.copy()
#         j_inp[j_inp == UNKNOWN_LOSS] = KNOWN
        
#         ax_x_inp_s = ax_row[3 + 4*i+3]
#         drawing_fn(
#             x_inp_s,
#             j=j_inp,
#             ax=ax_x_inp_s
#         )
#         ax_x_inp_s.set_title(f"x_inp_s{i}")

    for i, a_ in enumerate(a):
        for j, a_l in enumerate(a_):
            offset = 3 + 2*m.shape[0] + a.shape[1] * i + j
            ax_a_l = ax_row[offset]
            drawing_fn(
                a_l.reshape(*x.shape) + (1/2),
                ax=ax_a_l,
                clip=False
            )
            ttl = f"a_{i}_{j} "
#             ax_a_l.set_title(ttl + "m = {0:.2f}".format(np.mean(a_l)))

    for i, d_ in enumerate(d):
        offset = 3 + 2*m.shape[0] + a.shape[0] * a.shape[1] + i
        ax_d = ax_row[offset]
        drawing_fn(
            d_.reshape(*x.shape),
            ax=ax_d
        )
        ttl = f"d_{i} "
#         ax_d.set_title(ttl + "m = {0:.2f}".format(np.mean(d_)))
    for ax in ax_row:
        ax.axis("off")


def visualize_n_samples(
    x: np.ndarray, j: np.ndarray, p: np.ndarray, m: np.ndarray, a: np.ndarray, d: np.ndarray, y: int,
    title_prefixes: Dict[int, str] = None, ax_row: Optional[np.ndarray] = None,
    drawing_fn: Callable[[np.ndarray, np.ndarray, plt.Axes], None] = vis_digit_mask,
    sample_fn=gans_gmms_sample_no_d,
    n_samples: int = 10,
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
#     row_len = row_length(x, j, p, m, a, d, y)
#     if ax_row is None:
#         fig, ax_row = plt.subplots(1, row_len)

#     assert ax_row.shape[0] >= row_len
    if title_prefixes is None:
        title_prefixes = dict()

    title_prefixes = defaultdict(str, title_prefixes)
    ax_x_original = ax_row[0]


    drawing_fn(x, ax=ax_x_original)
    ax_x_original.set_title(f"{title_prefixes[0]} original")

    ax_x_masked = ax_row[1]
    drawing_fn(x, j, ax_x_masked)
    ax_x_masked.set_title(title_prefixes[1])

    ax_x_input = ax_row[2]
    drawing_fn(
        (x * (j == KNOWN)),
        ax=ax_x_input
    )

    ax_x_input.set_title("input to model")

    for i, m_ in enumerate(m):
        ax_m = ax_row[3 + 4*i]
        drawing_fn(
            m_.reshape(*x.shape),
            ax=ax_m
        )
        p_form = int(p[i] * 100) / 100
        ax_m.set_title(f"m_{i}, p={p_form}")
        
        ax_x_inp_m = ax_row[3 + 4*i+1]
        m_inp = m_.reshape(x.shape)
        x_inp = inpainted(x, j, m_inp)
        j_inp = j.copy()
        j_inp[j_inp==UNKNOWN_LOSS] = KNOWN
        drawing_fn(
            x_inp,
            j_inp,
            ax_x_inp_m
        )
        ax_x_inp_m.set_title(f"x_inp_m_{i}")
        a_ = a[i]
        d_ = d[i]
        
        for s_i in range(n_samples):
            s = sample_fn(x, m_, a_, d_)
            ax_s= ax_row[3 + 4*i+2 + s_i]

            drawing_fn(
                s,
                ax=ax_s
            )
            ax_s.set_title(f"s_{s_i}")

#             x_inp_s = inpainted(x, j, s)
#             j_inp = j.copy()
#             j_inp[j_inp == UNKNOWN_LOSS] = KNOWN

#             ax_x_inp_s = ax_row[3 + 4*i+3 + 2*s_i]
#             drawing_fn(
#                 x_inp_s,
#                 j=j_inp,
#                 ax=ax_x_inp_s
#             )
#             ax_x_inp_s.set_title(f"x_inp_s{s_i}")

