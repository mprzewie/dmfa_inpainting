from typing import Dict, List, Callable, Tuple

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from inpainting.datasets.mask_coding import KNOWN
from inpainting.utils import inpainted
from inpainting.visualizations.visualizations_utils import gans_gmms_sample_no_d
from inpainting import losses2 as l2
import torch


def outputs_to_images(
    x: np.ndarray,
    j: np.ndarray,
    p: np.ndarray,
    m: np.ndarray,
    a: np.ndarray,
    d: np.ndarray,
    y: np.ndarray,
):
    """
    Args:
        x: [c, h, w]
        j: [h, w]
        p: [mx]
        m: [mx, c*h*w]
        a: [mx, aw, c*h*w]
        d: [mx, c*h*w]
        y: []

    Returns:
    """

    original = x.transpose((1, 2, 0))
    mask = j.transpose((1, 2, 0))
    masked = original * (mask == KNOWN)
    means = [m_.reshape(x.shape).transpose((1, 2, 0)) for m_ in m]
    inpainted_with_means = [inpainted(original, mask, m_) for m_ in means]
    samples = [
        gans_gmms_sample_no_d(x, m_, a_, d_).reshape(x.shape).transpose(1, 2, 0)
        for (m_, a_, d_) in zip(m, a, d)
    ]
    inpainted_with_samples = [inpainted(original, mask, s_) for s_ in samples]
    return {
        "original": original,
        "mask": mask,
        "masked": masked,
        **{f"means_{i}": m_ for (i, m_) in enumerate(means)},
        **{f"inpainted_means_{i}": m_ for (i, m_) in enumerate(inpainted_with_means)},
        **{f"samples_{i}": s_ for (i, s_) in enumerate(samples)},
        **{
            f"inpainted_samples_{i}": s_
            for (i, s_) in enumerate(inpainted_with_samples)
        },
    }


MNIST_metrics = {
    "structural_similarity": lambda i1, i2: structural_similarity(
        i1, i2, multichannel=True
    ),
    "peak_signal_noise_ratio": lambda i1, i2,: peak_signal_noise_ratio(i1, i2),
}


def images_metrics(
    img_dict: Dict[str, np.ndarray],
    metrics_fns: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = MNIST_metrics,
) -> List[Dict]:
    original = img_dict["original"]
    return [
        {
            "img_kind": k,
            **{m_name: m(original, img) for (m_name, m) in metrics_fns.items()},
        }
        for (k, img) in img_dict.items()
    ]


def loss_like_metrics(
    model_outputs: Tuple[np.ndarray, ...],
    loss_fns: Dict[str, l2.InpainterLossFn] = dict(
        nll=l2.nll_buffered,
        mse=l2.mse_buffered,
        signed_diff_mean=l2.signed_difference_mean_buffered,
        signed_diff_std=l2.signed_difference_std_buffered,
    ),
):
    x, j, p, m, a, d, y = [
        torch.from_numpy(t if not isinstance(t, tuple) else t[0]).unsqueeze(0)
        for t in model_outputs
    ]
    return {fn_name: fn(x, j, p, m, a, d).item() for fn_name, fn in loss_fns.items()}
