from typing import Sequence, Callable, Tuple

import numpy as np
import torch

from inpainting.datasets.utils import RandomRectangleMaskConfig


def random_mask_fn(mask_configs: Sequence[RandomRectangleMaskConfig]) -> Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def tensor_to_tensor_with_random_mask(image_tensor: torch.Tensor):
        mask = np.ones((3, *image_tensor.shape[1:3]))
        for mc in mask_configs:
            mask = mc.generate_on_mask(mask)
        return image_tensor, torch.tensor(mask).float()

    return tensor_to_tensor_with_random_mask