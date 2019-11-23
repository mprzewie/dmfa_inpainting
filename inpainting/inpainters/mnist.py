import torch
from torch import nn

from inpainting.custom_layers import Reshape
from inpainting.inpainters.inpainter import InpainterModule


class MNISTInpainter(
    InpainterModule
):
    def __init__(self, n_mixes: int = 1, in_size: int = 784, a_width: int=3, hidden_size=1024):
        super().__init__()


        self.extractor = nn.Sequential(
            Reshape((-1, in_size * 2)),
            nn.Linear(in_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.a_extractor = nn.Sequential(
            nn.Linear(hidden_size, in_size * n_mixes * a_width),
            Reshape((-1, n_mixes, a_width, in_size,))  # * L, we don't want 1x4 vector but L x4 matrix))
        )
        self.m_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size)),
            nn.Sigmoid()

        )

        self.d_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size))

        )

        self.p_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes),
            nn.Softmax()
        )

    def forward(self, X, J):
        X_masked = X * J
        X_J = torch.cat([X_masked, J], dim=1)
        features = self.extractor(X_J)
        m = self.m_extractor(features)
        d = self.d_extractor(features)
        p = self.p_extractor(features)
        a = self.a_extractor(features)

        return p, m, a, d