from typing import Tuple

import torch
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


def train_val_datasets(mask_size: int, ) -> Tuple[TensorDataset, TensorDataset]:
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
    X.shape, J.shape, y.shape, set(y)
    X_train, X_val, J_train, J_val, y_train, y_val = train_test_split(X, J, y, test_size=0.33, random_state=42)
    ds_train = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(J_train),
        torch.tensor(y_train).long()
    )

    ds_val = TensorDataset(
        torch.tensor(X_val),
        torch.tensor(J_val),
        torch.tensor(y_val).long()
    )
    return ds_train, ds_val