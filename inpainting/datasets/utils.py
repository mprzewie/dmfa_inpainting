import dataclasses as dc
from typing import Sequence, Optional

import numpy as np
import torch
from inpainting.datasets import mask_coding as mc

@dc.dataclass(frozen=True)
class RandomMaskConfig:
    """A config for generating random masks"""

    value: int
    deterministic: bool
    size: int

    def generate_on_mask(
            self, mask: np.ndarray, copy: bool = True, seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Args:
            mask: [c, h, w]
            copy:
        """
        return mask


@dc.dataclass(frozen=True)
class RandomRectangleMaskConfig(RandomMaskConfig):
    height_ampl: int = 0
    width_ampl: int = 0

    def generate_on_mask(
            self, mask: np.ndarray, copy: bool = True, seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        mask = mask.copy() if copy else mask
        m_height = self.size + np.random.randint(
            -self.height_ampl, self.height_ampl + 1
        )
        m_width = self.size + np.random.randint(-self.width_ampl, self.width_ampl + 1)
        tot_height, tot_width = mask.shape[1:3]

        m_y = (
            np.random.randint(0, tot_height - m_height) if m_height < tot_height else 0
        )
        m_x = np.random.randint(0, tot_width - m_width) if m_width < tot_width else 0
        mask[..., m_y: m_y + m_height, m_x: m_x + m_width] = self.value
        return mask


@dc.dataclass(frozen=True)
class RandomNoiseMaskConfig(RandomMaskConfig):

    def generate_on_mask(
            self, mask: np.ndarray, copy: bool = True, seed: Optional[int] = None
    ) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        mask = mask.copy() if copy else mask
        c, h, w = mask.shape
        non_mask_size = h*w - self.size

        non_masked = [0] * non_mask_size
        masked = [1] * self.size
        mask_1d = non_masked + masked

        original_mask = mask[0].reshape(h*w) # first channel
        mask_perm = np.random.permutation(mask_1d)

        original_mask[mask_perm==1] = self.value
        mask = original_mask.reshape(h, w)
        return np.array([mask] * c)









def random_mask_fn(
        mask_configs: Sequence[RandomMaskConfig],  # deterministic: bool = True
):
    def tensor_to_tensor_with_random_mask(image_tensor: torch.Tensor):
        mask = np.ones_like(image_tensor.numpy())
        for mc in mask_configs:
            mask = mc.generate_on_mask(
                mask,
                seed=mc.value + int((image_tensor * 255).sum().item())
                if mc.deterministic
                else None,
            )
        return image_tensor, torch.tensor(mask).float()

    return tensor_to_tensor_with_random_mask
