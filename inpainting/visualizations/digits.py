from typing import Optional

import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt


def digit_with_mask(
        x: np.ndarray,
        j: np.ndarray,
        j2: np.ndarray,
        ax: Optional[Axis] = None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    x_j = x * j * j2
    j2_not_j = ((j2 + (j == 0)) == 0).astype(int)

    vis = np.vstack([x_j + j2_not_j, x_j, x_j + (j == 0)]).transpose((1, 2, 0))
    ax.imshow(vis, vmin=0, vmax=1)
    ax.axis("off")
    return ax


def model_input(
        x: np.ndarray,
        j: np.ndarray,
        j2: np.ndarray,
        ax: Optional[Axis] = None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    j_unified = j * j2
    x = x.copy()
    x[j_unified == 0] = 0 #x[j_unified == 1].mean()
    ax.imshow(x.squeeze(), vmin=0, vmax=1, cmap="gray")
    ax.axis("off")
    return ax
