import torch
from torch import nn

from inpainting.custom_layers import Reshape, LambdaLayer
from inpainting.inpainters.inpainter import InpainterModule


class RGBInpainter(
    InpainterModule
):
    def __init__(
        self,
        n_mixes: int = 1,
        h: int = 32,
        w: int = 32,
        last_channels: int = 12,
        a_width: int = 3,
        a_amplitude: float = 2
        ):
        super().__init__()
        c = 3
        in_size = c * h * w
        hidden_size = h * w * last_channels
        self.a_amplitude = a_amplitude

        self.extractor = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, last_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            Reshape((-1, hidden_size))
        )

        self.a_extractor = nn.Sequential(
            nn.Linear(hidden_size, in_size * n_mixes * a_width),
            Reshape((-1, n_mixes, a_width, in_size,)),
            LambdaLayer(self.postprocess_a)
        )
        self.m_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size)),

        )

        self.d_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size)),
            LambdaLayer(self.postprocess_d),
        )

        self.p_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes),
            nn.Softmax()
        )

    def postprocess_d(self, d_tensor: torch.Tensor):
        return torch.sigmoid(d_tensor) + 1e-10

    def postprocess_a(self, a_tensor: torch.Tensor):
        return self.a_amplitude * torch.sigmoid(a_tensor) - (self.a_amplitude / 2)

    def forward(self, X, J):
        X_masked = X * J
        X_J = torch.cat([X_masked, J], dim=1)

        features = self.extractor(X_J)

        m = self.m_extractor(features)
        d = self.d_extractor(features)
        p = self.p_extractor(features)
        a = self.a_extractor(features)

        return p, m, a, d