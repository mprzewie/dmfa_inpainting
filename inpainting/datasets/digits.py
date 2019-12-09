from typing import Tuple

import torch
from cached_property import cached_property
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data.dataset import Dataset
import dataclasses as dc


@dc.dataclass(frozen=True)
class DigitsDataset(Dataset):
    X: np.ndarray
    J: np.ndarray
    y: np.ndarray

    def __post_init__(self):
        assert self.X.shape[0] == self.J.shape[0] == self.y.shape[0]

    @cached_property
    def X_tensor(self) -> torch.Tensor:
        return torch.tensor(self.X).float()

    @cached_property
    def J_tensor(self) -> torch.Tensor:
        return torch.tensor(self.J).float()

    @cached_property
    def y_tensor(self) -> torch.Tensor:
        return torch.tensor(self.y)

    def __getitem__(self, item):
        return (self.X_tensor[item], self.J_tensor[item]), self.y_tensor[item]

    def __len__(self):
        return self.X.shape[0]


def train_val_datasets(mask_size: int, ) -> Tuple[Dataset, Dataset]:
    digits = datasets.load_digits()
    X = digits['data']
    y = digits['target']
    J = []
    for i in range(X.shape[0]):
        mask = np.ones((8, 8))
        m_height = mask_size
        m_width = mask_size
        m_x = np.random.randint(0, 8 - m_width)
        m_y = np.random.randint(0, 8 - m_height)

        mask[m_y:m_y + m_height, m_x:m_x + m_width] = 0
        J.append(mask.reshape(-1))

    J = np.vstack(J)
    X = X / 16

    X = X.reshape(-1, 1, 8, 8)
    J = J.reshape(-1, 1, 8, 8)
    X_train, X_val, J_train, J_val, y_train, y_val = train_test_split(X, J, y, test_size=0.33, random_state=42)

    ds_train = DigitsDataset(X_train, J_train, y_train)
    ds_val = DigitsDataset(X_val, J_val, y_val)

    return ds_train, ds_val
