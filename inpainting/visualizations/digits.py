from typing import Optional

import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt

from inpainting.datasets.mask_coding import KNOWN, UNKNOWN_LOSS, UNKNOWN_NO_LOSS


def digit_with_mask(
        x: np.ndarray,
        j: np.ndarray,
        ax: Optional[Axis] = None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    x_j = x * (j == KNOWN)
    vis = np.vstack([
        x_j + (j == UNKNOWN_NO_LOSS),
        x_j,
        x_j + (j == UNKNOWN_LOSS)
    ]).transpose((1, 2, 0))
    ax.imshow(vis, vmin=0, vmax=1, cmap="gray")
    ax.axis("off")
    return ax

