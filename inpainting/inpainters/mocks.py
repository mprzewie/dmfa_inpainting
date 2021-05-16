import torch

from inpainting.datasets.mask_coding import KNOWN
from inpainting.inpainters.inpainter import InpainterModule
from torch.utils.data import Dataset, DataLoader
from sklearn.impute import KNNImputer
from typing import Optional
import numpy as np
from time import time
from tqdm import tqdm


class ZeroInpainter(InpainterModule):
    def __init__(self, n_mixes: int = 1, a_width: int = 1):
        super().__init__(n_mixes=n_mixes, a_width=a_width)

    def forward(self, X: torch.Tensor, J: torch.Tensor):
        b, c, h, w = X.shape
        chw = c * h * w
        P = torch.zeros((b, self.n_mixes))
        P[:, 0] = 1  # GT returned in the first mixture
        M = torch.zeros((b, self.n_mixes, chw))
        A = torch.zeros((b, self.n_mixes, self.a_width, chw))  # 0 variance
        D = (
            torch.ones(b, self.n_mixes, chw) * 1e-6
        )  # minimal noise where X is unknown, 0 where X is known
        P, M, A, D = [t.to(X.device) for t in [P, M, A, D]]
        return P, M, A, D


class GroundTruthInpainter(ZeroInpainter):
    def forward(self, X: torch.Tensor, J: torch.Tensor):
        P, M, A, D = super().forward(X, J)
        X_unknown = X * (J != KNOWN)  # return gt of only unknown parts
        b, n, chw = M.shape
        M[:, 0] = X_unknown.reshape((b, chw))
        return P, M, A, D


class NoiseInpainter(ZeroInpainter):
    def forward(self, X: torch.Tensor, J: torch.Tensor):
        P, M, A, D = super().forward(X, J)
        noise = torch.randn_like(X) * (J != KNOWN)  # return gt of only unknown parts
        b, n, chw = M.shape
        M[:, 0] = noise.reshape((b, chw))
        return P, M, A, D


class KNNInpainter(ZeroInpainter):
    def __init__(
        self,
        ds_train: Dataset,
        backend: str = "cuml",
        n_mixes: int = 1,
        a_width: int = 1,
        max_samples: int = 5000,
    ):
        super().__init__(n_mixes=n_mixes, a_width=a_width)
        self.backend = backend
        self.max_samples = max_samples
        print(f"Using KNN backend: {backend}")
        if backend == "sklearn":
            self._init_sklearn(ds_train)

        elif backend == "cuml":
            self._init_cuml(ds_train)
        else:
            raise TypeError("Unknown KNN backend!")

    def forward(self, X: torch.Tensor, J: torch.Tensor):
        if self.backend == "sklearn":
            return self._forward_sklearn(X, J)
        else:
            return self._forward_cuml(X, J)

    def _init_cuml(self, ds_train):
        import cuml, cudf

        dl = DataLoader(ds_train, shuffle=True, batch_size=self.max_samples)

        for (X, J), y in tqdm(dl, "Fitting CUML KNN"):
            break

        self.data = X
        self.data_mean = self.data.mean(dim=0)
        print(self.data_mean.shape)
        data_np = X.numpy()
        knn = cuml.neighbors.NearestNeighbors().fit(data_np.reshape(len(data_np), -1))
        self.knn = knn

    def _forward_cuml(self, X: torch.Tensor, J: torch.Tensor):
        import cuml, cudf

        P, M, A, D = super().forward(X, J)

        XJ = (X * (J == KNOWN)) + (self.data_mean.to(J.device) * (J != KNOWN))
        _, indices = self.knn.kneighbors(
            XJ.reshape(XJ.shape[0], -1).detach().cpu().numpy(), n_neighbors=1
        )

        X_inp = self.data[torch.tensor(indices)]
        b, n, chw = M.shape
        M[:, 0] = X_inp.reshape(b, chw).to(M.device)
        return P, M, A, D

    def _init_sklearn(
        self,
        ds_train,
    ):
        knn = KNNImputer()
        data_np = np.array([X.numpy() for (X, J), y in ds_train])
        data_np = np.random.choice(data_np, self.max_samples)
        data_np = data_np.reshape(len(data_np), -1)

        print(f"Fitting {knn} to {len(data_np)} examples...")
        knn.fit(data_np)
        self.knn = knn

    def _forward_sklearn(self, X: torch.Tensor, J: torch.Tensor):
        P, M, A, D = super().forward(X, J)

        X_np = X.cpu().reshape(len(X), -1).numpy()
        J_np = J.cpu().reshape(len(J), -1).numpy()
        X_np[J_np != KNOWN] = np.nan

        X_inp = self.knn.transform(X_np)
        b, n, chw = M.shape
        M[:, 0] = torch.tensor(X_inp.reshape((b, chw))).to(M.device)
        return P, M, A, D
