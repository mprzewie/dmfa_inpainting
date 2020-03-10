from pathlib import Path
from typing import Tuple, List, Sequence, TypeVar

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms as tr

from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS, KNOWN
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
        mask = np.ones_like(image_tensor.numpy())
        for mc in mask_configs:
            mask = mc.generate_on_mask(mask)
        return image_tensor, torch.tensor(mask).float()

    return tensor_to_tensor_with_random_mask


def train_val_datasets(
        save_path: Path,
        mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_MASK_CONFIGS,
        ds_type:  MNIST = MNIST
) -> Tuple[MNIST, MNIST]:
    train_transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(random_mask_fn(mask_configs=mask_configs))
    ])

    val_transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(random_mask_fn(
            mask_configs=[
                m for m in mask_configs if m.value==UNKNOWN_LOSS or m.value == KNOWN
            ] # only the mask which will be inpainted
        ))
    ])

    ds_train = ds_type(save_path, train=True, download=True, transform=train_transform)
    ds_val = ds_type(save_path, train=False, download=True, transform=val_transform)

    return ds_train, ds_val
