import dataclasses as dc

import numpy as np


@dc.dataclass(frozen=True)
class RandomRectangleMaskConfig:
    """A config for generating random masks"""
    value: int
    height: int
    width: int
    height_ampl: int = 0
    width_ampl: int = 0

    def generate_on_mask(self, mask: np.ndarray, copy: bool = True, seed: int = None) -> np.ndarray:
        """
        Args:
            mask: [h, w]
            copy:
        """
        if seed is not None:
            np.random.seed(seed)
        mask = mask.copy() if copy else mask
        m_height = self.height + np.random.randint(-self.height_ampl, self.height_ampl + 1)
        m_width = self.width + np.random.randint(-self.width_ampl, self.width_ampl + 1)
        tot_height, tot_width = mask.shape[1:3]
        
        m_y = np.random.randint(0, tot_height - m_height) if m_height < tot_height else 0
        m_x = np.random.randint(0, tot_width- m_width) if m_width < tot_width else 0
        mask[..., m_y:m_y + m_height, m_x:m_x + m_width] = self.value
        return mask
