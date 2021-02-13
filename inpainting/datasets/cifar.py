from pathlib import Path
from typing import Tuple, Sequence, Type

from torchvision import transforms as tr
from torchvision.datasets import CIFAR10

from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS
from inpainting.datasets.rgb_utils import random_mask_fn
from inpainting.datasets.utils import RandomRectangleMaskConfig

DEFAULT_MASK_CONFIGS = (
    RandomRectangleMaskConfig(
        UNKNOWN_LOSS,
        15,
        15,
        0,
        0,
    ),
    RandomRectangleMaskConfig(UNKNOWN_NO_LOSS, 15, 15, 2, 2),
)


def train_val_datasets(
    save_path: Path,
    mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_MASK_CONFIGS,
    resize_size: Tuple[int, int] = (32, 32),
    ds_cls: Type[CIFAR10] = CIFAR10,
) -> Tuple[CIFAR10, CIFAR10]:
    train_transform = tr.Compose(
        [
            tr.Resize(resize_size),
            tr.ToTensor(),
            tr.Lambda(random_mask_fn(mask_configs=mask_configs)),
        ]
    )

    val_transform = tr.Compose(
        [
            tr.ToTensor(),
            tr.Lambda(
                random_mask_fn(
                    mask_configs=[
                        m for m in mask_configs if m.value == UNKNOWN_LOSS
                    ]  # only the mask which will be inpainted
                )
            ),
        ]
    )

    ds_train = ds_cls(save_path, train=True, download=True, transform=train_transform)
    ds_val = ds_cls(save_path, train=False, download=True, transform=val_transform)

    return ds_train, ds_val
