import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from inpainting.datasets.mask_coding import KNOWN
from inpainting.utils import inpainted
from inpainting.visualizations.samples import gans_gmms_sample_no_d


def outputs_to_images(
        x: np.ndarray,
        j: np.ndarray,
        p: np.ndarray,
        m: np.ndarray,
        a: np.ndarray,
        d: np.ndarray,
        y: np.ndarray
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
    mask = j.transpose((1,2, 0))
    masked = original * (mask == KNOWN)
    means = [
        m_.reshape(x.shape).transpose((1, 2, 0))
        for m_ in m
    ]
    inpainted_with_means = [
        inpainted(
            original, mask, m_
        )
        for m_ in means
    ]
    samples = [
        gans_gmms_sample_no_d(
            x, m_, a_, d_
        ).reshape(x.shape).transpose(1, 2, 0)
        for (m_, a_, d_) in zip(m, a, d)
    ]
    inpainted_with_samples = [
        inpainted(original, mask, s_)
        for s_ in samples
    ]
    return {
        "original": original,
        "mask": mask,
        "masked": masked,
        **{
            f"means_{i}": m_
            for (i, m_) in enumerate(means)
          },
        **{
            f"inpainted_means_{i}": m_
            for (i, m_) in enumerate(inpainted_with_means)
        },
        **{
            f"samples_{i}": s_
            for (i, s_) in enumerate(samples)
        },
        **{
            f"inpainted_samples_{i}": s_
            for (i, s_) in enumerate(inpainted_with_samples)
        }

    }
