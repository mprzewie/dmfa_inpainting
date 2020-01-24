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
    J_2: np.ndarray
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
    def J_2_tensor(self) -> torch.Tensor:
        return torch.tensor(self.J_2).float()

    @cached_property
    def y_tensor(self) -> torch.Tensor:
        return torch.tensor(self.y)

    def __getitem__(self, item):
        return (self.X_tensor[item], self.J_tensor[item], self.J_2_tensor[item]), self.y_tensor[item]

    def __len__(self):
        return self.X.shape[0]


def train_val_datasets(mask_size: int = 3, mask_2_size: int = 2, mask_variance: int = 0, mask_2_variance: int = 0) -> \
Tuple[Dataset, Dataset]:
    digits = datasets.load_digits()
    X = digits['data']
    y = digits['target']
    J = []
    J_2 = []
    for i in range(X.shape[0]):
        # mask which model will train on
        mask = np.ones((8, 8))
        m_height = mask_size + np.random.randint(-mask_variance, mask_variance + 1)
        m_width = mask_size + np.random.randint(-mask_variance, mask_variance + 1)
        m_x = np.random.randint(0, 8 - m_width)
        m_y = np.random.randint(0, 8 - m_height)

        mask[m_y:m_y + m_height, m_x:m_x + m_width] = 0
        mask_flat = mask.reshape(-1)

        # mask with unknown data
        mask_2 = np.ones((8, 8))
        m_2_height = mask_2_size + np.random.randint(-mask_2_variance, mask_2_variance + 1)
        m_2_width = mask_2_size + np.random.randint(-mask_2_variance, mask_2_variance + 1)
        m_2_x = np.random.randint(0, 8 - m_width)
        m_2_y = np.random.randint(0, 8 - m_height)
        mask_2[m_2_y:m_2_y + m_2_height, m_2_x:m_2_x + m_2_width] = 0
        mask_2_flat = mask_2.reshape(-1)

        mask_flat[mask_2_flat == 0] = 1
        # mask 1 cannot be 0 where mask 2 is 0
        # so, if data is truly unknown (as specified by mask 2)
        # we cannot calculate loss from it (as specified by mask 1)

        J.append(mask_flat)
        J_2.append(mask_2_flat)

    J = np.vstack(J)
    J_2 = np.vstack(J_2)
    X = X / 16

    X = X.reshape(-1, 1, 8, 8)
    J = J.reshape(-1, 1, 8, 8)
    J_2 = J_2.reshape(-1, 1, 8, 8)
    X_train, X_val, J_train, J_val, J_2_train, J_2_val, y_train, y_val = train_test_split(X, J, J_2, y, test_size=0.33,
                                                                                          random_state=42)

    ds_train = DigitsDataset(X_train, J_train, J_2_train, y_train)
    ds_val = DigitsDataset(X_val, J_val, J_2_val, y_val)

    return ds_train, ds_val

import jpredapi
jpredapi.status
