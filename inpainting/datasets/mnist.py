from pathlib import Path
from typing import Tuple, Sequence, Type

from torchvision import transforms as tr
from torchvision.datasets import MNIST

from inpainting.datasets.utils import random_mask_fn, RandomMaskConfig


def train_val_datasets(
    save_path: Path,
    mask_configs_train: Sequence[RandomMaskConfig],
    mask_configs_val: Sequence[RandomMaskConfig],
    ds_type: Type[MNIST] = MNIST,
    resize_size: Tuple[int, int] = (28, 28),
) -> Tuple[MNIST, MNIST]:

    base_transform = tr.Compose([tr.Resize(resize_size), tr.ToTensor()])
    train_transform = tr.Compose(
        [
            base_transform,
            tr.Lambda(random_mask_fn(mask_configs=mask_configs_train)),
        ]
    )

    val_transform = tr.Compose(
        [
            base_transform,
            tr.Lambda(random_mask_fn(mask_configs=mask_configs_val)),
        ]
    )

    ds_train = ds_type(save_path, train=True, download=True, transform=train_transform)
    ds_val = ds_type(save_path, train=False, download=True, transform=val_transform)

    return ds_train, ds_val
