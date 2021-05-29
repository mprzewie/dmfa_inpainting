from pathlib import Path
from typing import Tuple, Sequence

from torchvision import transforms as tr
from torchvision.datasets import CelebA

from inpainting.datasets.utils import RandomRectangleMaskConfig, random_mask_fn


def train_val_datasets(
    save_path: Path,
    mask_configs_train: Sequence[RandomRectangleMaskConfig],
    mask_configs_val: Sequence[RandomRectangleMaskConfig],
    resize_size: Tuple[int, int] = (50, 50),
    crop_size: Tuple[int, int] = (32, 32),
) -> Tuple[CelebA, CelebA]:

    base_transform = tr.Compose(
        [
            tr.Lambda(lambda im: im.convert("RGB")),
            tr.Resize(resize_size),
            tr.CenterCrop(crop_size),
            tr.ToTensor(),
        ]
    )

    train_transform = tr.Compose(
        [
            base_transform,
            tr.Lambda(random_mask_fn(mask_configs=mask_configs_train)),
        ]
    )

    val_transform = tr.Compose(
        [
            base_transform,
            tr.Lambda(
                random_mask_fn(
                    mask_configs=mask_configs_val,
                )
            ),
        ]
    )

    ds_train = CelebA(
        save_path, split="train", download=False, transform=train_transform
    )
    ds_val = CelebA(save_path, split="valid", download=False, transform=val_transform)

    return ds_train, ds_val
