from typing import Optional

import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt

from inpainting.datasets.mask_coding import KNOWN, UNKNOWN_LOSS, UNKNOWN_NO_LOSS


def digit_with_mask(
        x: np.ndarray,
        j: np.ndarray = None,
        ax: Optional[Axis] = None,
        clip: bool = True

):
    j = np.ones_like(x) * KNOWN if j is None else j

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x_j = x * (j == KNOWN)
    vis = np.vstack([
        x_j + (j == UNKNOWN_NO_LOSS),
        x_j,
        x_j + (j == UNKNOWN_LOSS)
    ]).transpose((1, 2, 0))
    
    vis = np.clip(vis, 0, 1) if clip else vis
    ax.imshow(vis, vmin=0, vmax=1, cmap="gray")
    ax.axis("off")
    return ax


def rgb_with_mask(
        x: np.ndarray,
        j: np.ndarray = None,
        ax: Optional[Axis] = None,
        clip: bool = True
):
    j = np.ones_like(x) * KNOWN if j is None else j
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x_j = x * (j == KNOWN)

    x_j[2][j[2]==UNKNOWN_LOSS] = 1
    x_j[0][j[0]==UNKNOWN_NO_LOSS] = 1

    x_j = x_j.transpose((1, 2, 0))
    
    x_j = np.clip(x_j, 0, 1) if clip else x_j
    ax.imshow(
        x_j
    ) #, cmap="gray")
    ax.axis("off")
    return ax



