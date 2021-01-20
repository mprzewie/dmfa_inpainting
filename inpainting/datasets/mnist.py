from pathlib import Path
from typing import Tuple, Sequence, Type

from torchvision.datasets import MNIST
from torchvision import transforms as tr
from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS, KNOWN
from inpainting.datasets.utils import RandomRectangleMaskConfig, random_mask_fn

DEFAULT_MASK_CONFIGS = (
    RandomRectangleMaskConfig(UNKNOWN_LOSS, 8, 8, 2, 2),
    RandomRectangleMaskConfig(UNKNOWN_NO_LOSS, 8, 8, 2, 2),
)


def train_val_datasets(
    save_path: Path,
    mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_MASK_CONFIGS,
    ds_type: Type[MNIST] = MNIST,
    resize_size: Tuple[int, int] = (28, 28),
) -> Tuple[MNIST, MNIST]:

    base_transform = tr.Compose([tr.Resize(resize_size), tr.ToTensor()])
    train_transform = tr.Compose(
        [
            base_transform,
            tr.Lambda(random_mask_fn(mask_configs=mask_configs, deterministic=False)),
        ]
    )

    val_transform = tr.Compose(
        [
            base_transform,
            tr.Lambda(
                random_mask_fn(
                    mask_configs=[
                        m for m in mask_configs if m.value in [UNKNOWN_LOSS, KNOWN]
                    ],  # only the mask which will be inpainted
                    deterministic=False,
                )
            ),
        ]
    )

    ds_train = ds_type(save_path, train=True, download=True, transform=train_transform)
    ds_val = ds_type(save_path, train=False, download=True, transform=val_transform)

    return ds_train, ds_val
