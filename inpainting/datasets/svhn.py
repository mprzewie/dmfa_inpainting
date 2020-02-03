from pathlib import Path
from typing import Tuple, Sequence

from torchvision import transforms as tr
from torchvision.datasets import SVHN

from inpainting.datasets.mask_coding import UNKNOWN_LOSS
from inpainting.datasets.rgb_utils import random_mask_fn
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


def train_val_datasets(
        save_path: Path,
        mask_configs: Sequence[RandomRectangleMaskConfig] = DEFAULT_MASK_CONFIGS,
) -> Tuple[SVHN, SVHN]:
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(random_mask_fn(
            mask_configs=[
                m for m in mask_configs if m.value == UNKNOWN_LOSS
            ]  # only the mask which will be inpainted
        ))
    ])

    ds_train = SVHN(save_path, split="train", download=True, transform=transform)
    ds_val = SVHN(save_path, split="test", download=True, transform=transform)

    return ds_train, ds_val