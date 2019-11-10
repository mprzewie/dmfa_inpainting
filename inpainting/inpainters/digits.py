import torch
from torch import nn

from inpainting.custom_layers import Reshape
from inpainting.inpainters.inpainter import InpainterModule


class DigitsLinearInpainter(
    InpainterModule
):
    def __init__(self, n_mixes: int = 1, a_width: int=3, hidden_size: int = 128):
        super().__init__(n_mixes=n_mixes, a_width=a_width)
        in_size = 64
        self.extractor = nn.Sequential(
            nn.Linear(in_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
            # nn.Linear(hidden_size, hidden_size),
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
            nn.Softmax(dim=-1)
        )

    def forward(self, X: torch.Tensor, J: torch.Tensor):
        X_masked = X * J
        X_J = torch.cat([X_masked, J], dim=1).float()
        features = self.extractor(X_J)
        m = self.m_extractor(features)
        d = self.d_extractor(features)
        p = self.p_extractor(features)
        a = self.a_extractor(features)

        return p, m, a, d