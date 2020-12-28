from typing import Tuple

import torch
from torch import nn

from inpainting.custom_layers import Reshape, LambdaLayer
from inpainting.datasets.mask_coding import KNOWN
from inpainting.inpainters.inpainter import InpainterModule


class LinearHeadsInpainter(InpainterModule):
    """
    linear_heads architecture.
    It is called this way, because it has a convolutional backbone and linear layers which predict outputs.
    """

    def __init__(
        self,
        n_mixes: int = 1,
        c_h_w: Tuple[int, int, int] = (1, 28, 28),
        last_channels: int = 12,
        a_width: int = 3,
        a_amplitude: float = 0.5,
    ):
        super().__init__(n_mixes=n_mixes, a_width=a_width)
        c, h, w = c_h_w
        in_size = c * h * w
        hidden_size = h * w * last_channels
        self.a_amplitude = a_amplitude

        self.extractor = nn.Sequential(
            nn.Conv2d(2 * c, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, last_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            Reshape((-1, hidden_size)),
        )

        self.a_extractor = nn.Sequential(
            nn.Linear(hidden_size, in_size * n_mixes * a_width),
            Reshape((-1, n_mixes, a_width, in_size)),
            LambdaLayer(self.postprocess_a),
        )
        self.m_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size), Reshape((-1, n_mixes, in_size))
        )

        self.d_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size)),
            LambdaLayer(self.postprocess_d),
        )

        self.p_extractor = nn.Sequential(nn.Linear(hidden_size, n_mixes), nn.Softmax())

    def postprocess_d(self, d_tensor: torch.Tensor):
        return torch.sigmoid(d_tensor) + 1e-10

    def postprocess_a(self, a_tensor: torch.Tensor):
        return self.a_amplitude * torch.sigmoid(a_tensor) - (self.a_amplitude / 2)

    def forward(self, X, J):
        J = J * (J == KNOWN)
        X_masked = X * J
        X_J = torch.cat([X_masked, J], dim=1)

        features = self.extractor(X_J)

        m = self.m_extractor(features)
        d = self.d_extractor(features)
        p = self.p_extractor(features)
        a = self.a_extractor(features)
        #         print(p.shape, m.shape, a.shape, d.shape)
        return p, m, a, d
