from typing import List, Callable

import numpy as np
from matplotlib import pyplot as plt


def plot_arrays_stats(
    arrays: List[np.ndarray],
    ax=None,
    stat_fns: List[Callable[[List], float]] = [np.min, np.max, np.mean],
    markers=".",
):
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(markers, str):
        markers = [markers] * len(stat_fns)

    for fn, m in zip(stat_fns, markers):
        ax.scatter(
            range(len(arrays)), [fn(a) for a in arrays], marker=m, label=fn.__name__
        )

    return ax
