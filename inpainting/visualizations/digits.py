from typing import Optional

import numpy as np
from matplotlib.axis import Axis
import matplotlib.pyplot as plt


def digit_with_mask(
        x: np.ndarray,
        j: np.ndarray,
        ax: Optional[Axis] = None
):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    x_j = x * j
    vis = np.vstack([x_j, x_j, x_j + (j == 0)]).T.reshape(8, 8, 3)

    ax.imshow(vis)
    ax.axis("off")
    return ax

