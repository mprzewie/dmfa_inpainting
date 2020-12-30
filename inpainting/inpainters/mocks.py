import torch

from inpainting.datasets.mask_coding import KNOWN
from inpainting.inpainters.inpainter import InpainterModule


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
