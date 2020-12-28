import torch

from datasets.mask_coding import KNOWN
from inpainters.inpainter import InpainterModule


class ZeroInpainter(InpainterModule):
    def forward(self, X: torch.Tensor, J: torch.Tensor):
        b, c, h, w = X.shape
        chw = c * h * w
        P = torch.zeros((b, self.n_mixes))
        P[:, 0] = 1  # GT returned in the first mixture
        M = torch.zeros((b, self.n_mixes, chw))
        A = torch.zeros((b, self.n_mixes, self.a_width, chw))  # 0 variance
        D = torch.zeros((b, self.n_mixes, chw))  # 0 noise
        P, M, A, D = [t.to(X.device) for t in [P, M, A, D]]
        return P, M, A, D


class GroundTruthInpainter(ZeroInpainter):
    def forward(self, X: torch.Tensor, J: torch.Tensor):
        P, M, A, D = super().forward(X, J)
        X_unknown = X * (J != KNOWN)  # return gt of only unknown parts
        b, n, chw = M.shape
        M[:, 0] = X_unknown.reshape(b, 1, chw)
        return P, M, A, D


class NoiseInpainter(ZeroInpainter):
    def forward(self, X: torch.Tensor, J: torch.Tensor):
        P, M, A, D = super().forward(X, J)
        noise = torch.randn_like(X) * (J != KNOWN)  # return gt of only unknown parts
        b, n, chw = M.shape
        M[:, 0] = noise.reshape(b, 1, chw)
        return P, M, A, D
