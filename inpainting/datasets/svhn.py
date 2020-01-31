from pathlib import Path
from typing import Tuple, Sequence

import numpy as np
import torch
from torchvision import transforms as tr
from torchvision.datasets import SVHN

from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS
from inpainting.datasets.utils import RandomRectangleMaskConfig

DEFAULT_MASK_CONFIGS = (
    RandomRectangleMaskConfig(
        UNKNOWN_LOSS,
        12, 12, 0,0,
    ),
    # RandomRectangleMaskConfig(
    #     UNKNOWN_NO_LOSS,
    #     8,8,2,2
    # )
)


def random_mask_fn(mask_configs: Sequence[RandomRectangleMaskConfig]):
    def tensor_to_tensor_with_random_mask(image_tensor: torch.Tensor):
        mask = np.ones((3, *image_tensor.shape[1:3]))
        for mc in mask_configs:
            mask = mc.generate_on_mask(mask)
        return image_tensor, torch.tensor(mask).float()

    return tensor_to_tensor_with_random_mask


def train_val_datasets(
        save_path: Path,
        mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_MASK_CONFIGS,
) -> Tuple[SVHN, SVHN]:
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(random_mask_fn(mask_configs=mask_configs))
    ])

    ds_train = SVHN(save_path, split="train", download=True, transform=transform)
    ds_val = SVHN(save_path, split="test", download=True, transform=transform)

    return ds_train, ds_val
