from typing import Tuple

import torch
from torch import nn

from inpainting.custom_layers import Reshape, LambdaLayer
from inpainting.datasets.mask_coding import KNOWN
from inpainting.inpainters.inpainter import InpainterModule


class FullyConvolutionalInpainter(InpainterModule):
    """fullconv architecture"""

    def __init__(
        self,
        extractor: nn.Module,
        n_mixes: int = 1,
        last_channels: int = 128,
        a_width: int = 3,
        a_amplitude: float = 2,
        c_h_w: Tuple[int, int, int] = (1, 28, 28),
        m_offset: float = 0,
    ):
        super().__init__(n_mixes=n_mixes, a_width=a_width)
        c, h, w = c_h_w
        in_size = c * h * w
        # hidden_size = h * w * last_channels
        self.a_amplitude = a_amplitude
        self.m_offset = m_offset
        self.extractor = extractor

        self.a_extractor = nn.Sequential(
            nn.Conv2d(last_channels, last_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm2d(last_channels // 2),
            nn.Conv2d(
                last_channels // 2, n_mixes * a_width * c, kernel_size=3, padding=1
            ),
            Reshape((-1, n_mixes, a_width, in_size)),
            LambdaLayer(self.postprocess_a),
        )
        self.m_extractor = nn.Sequential(
            nn.Conv2d(last_channels, last_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(last_channels // 2),
            nn.ReLU(),
            nn.Conv2d(last_channels // 2, n_mixes * c, kernel_size=3, padding=1),
            Reshape((-1, n_mixes, in_size)),
            LambdaLayer(self.postprocess_m),
        )

        self.d_extractor = nn.Sequential(
            nn.Conv2d(last_channels, n_mixes * c, kernel_size=3, padding=1),
            Reshape((-1, n_mixes, in_size)),
            LambdaLayer(self.postprocess_d),
        )

        self.p_extractor = nn.Sequential(
            nn.Conv2d(last_channels, 1, kernel_size=(h, w), padding=0),
            Reshape((-1, n_mixes)),
            nn.Softmax(),
        )

    def postprocess_d(self, d_tensor: torch.Tensor):
        return torch.sigmoid(d_tensor) + 1e-10

    def postprocess_a(self, a_tensor: torch.Tensor):
        return self.a_amplitude * torch.sigmoid(a_tensor) - (self.a_amplitude / 2)

    def postprocess_m(self, m_tensor: torch.Tensor):
        return m_tensor + self.m_offset

    def forward(self, X, J):
        J = J * (J == KNOWN)
        X_masked = X * J
        X_J = torch.cat([X_masked, J], dim=1)
        features = self.extractor(X_J)

        m = self.m_extractor(features)
        d = self.d_extractor(features)
        p = self.p_extractor(features)
        a = self.a_extractor(features)

        return p, m, a, d
