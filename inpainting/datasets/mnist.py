from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision import transforms as tr

def random_mask_fn(mask_shape: Tuple[int, int], mask_randomness: Tuple[int, int] = (0,0)):
    h, w = mask_shape
    hr, wr = mask_randomness

    def tensor_to_tensor_with_random_mask(image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_r = h + np.random.randint(-hr, hr)
        w_r = w + np.random.randint(-wr, wr)
        sx, sy = image_tensor.shape[1:]
        mask = np.ones((1, sx, sy))
        x, y = np.random.randint([0,0], [sx - h_r, sy - w_r])
        mask[0, x:x+h_r, y:y+w_r] = 0
        return (
            image_tensor,
            torch.tensor(mask).float(),
            torch.ones_like(image_tensor)
        )
    return tensor_to_tensor_with_random_mask


def train_val_datasets(save_path: Path, mask_shape: Tuple[int, int] = (10, 10), mask_randomness: Tuple[int, int] = (0,0)) -> Tuple[MNIST, MNIST]:
    transform = tr.Compose([
        tr.ToTensor(),
        tr.Lambda(random_mask_fn(mask_shape=mask_shape, mask_randomness=mask_randomness))
    ])

    ds_train = MNIST(save_path, train=True, download=True, transform=transform)
    ds_val = MNIST(save_path, train=False, download=True, transform=transform)

    return ds_train, ds_val