from pathlib import Path
from typing import Tuple, Sequence

from torchvision import transforms as tr
from torchvision.datasets import SVHN

from inpainting.datasets.utils import RandomRectangleMaskConfig, random_mask_fn


def train_val_datasets(
    save_path: Path,
    mask_configs_train: Sequence[RandomRectangleMaskConfig],
    mask_configs_val: Sequence[RandomRectangleMaskConfig],
    resize_size: Tuple[int, int] = (32, 32),
) -> Tuple[SVHN, SVHN]:
    train_transform = tr.Compose(
        [
            tr.Resize(resize_size),
            tr.ToTensor(),
            tr.Lambda(random_mask_fn(mask_configs=mask_configs_train)),
        ]
    )

    val_transform = tr.Compose(
        [
            tr.Resize(resize_size),
            tr.ToTensor(),
            tr.Lambda(random_mask_fn(mask_configs=mask_configs_val)),
        ]
    )

    ds_train = SVHN(save_path, split="train", download=True, transform=train_transform)
    ds_val = SVHN(save_path, split="test", download=True, transform=val_transform)

    return ds_train, ds_val
