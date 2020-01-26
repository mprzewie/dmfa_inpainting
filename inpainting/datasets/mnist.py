from pathlib import Path
from typing import Tuple, List, Sequence, TypeVar

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms as tr

from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS
from inpainting.datasets.utils import RandomRectangleMaskConfig

DEFAULT_MASK_CONFIGS = (
    RandomRectangleMaskConfig(
        UNKNOWN_LOSS,
        8,8,2,2
    ),
    RandomRectangleMaskConfig(
        UNKNOWN_NO_LOSS,
        8,8,2,2
    )
)


def random_mask_fn(mask_configs: Sequence[RandomRectangleMaskConfig]):
    def tensor_to_tensor_with_random_mask(image_tensor: torch.Tensor):
        mask = np.ones((image_tensor.shape[1:3]))
        for mc in mask_configs:
            mask = mc.generate_on_mask(mask.squeeze())
            mask = np.expand_dims(mask, 0)
        return image_tensor, torch.tensor(mask).float()

    return tensor_to_tensor_with_random_mask


def train_val_datasets(
        save_path: Path,
        mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_MASK_CONFIGS,
        ds_type:  MNIST = MNIST
) -> Tuple[MNIST, MNIST]:
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(random_mask_fn(mask_configs=mask_configs))
    ])

    ds_train = ds_type(save_path, train=True, download=True, transform=transform)
    ds_val = ds_type(save_path, train=False, download=True, transform=transform)

    return ds_train, ds_val
