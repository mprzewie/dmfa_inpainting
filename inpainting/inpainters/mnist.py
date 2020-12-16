"""Inpainters we tried out on MNIST, but didn't end up using."""
import torch
from torch import nn

from inpainting.custom_layers import Reshape, LambdaLayer
from inpainting.datasets.mask_coding import KNOWN
from inpainting.inpainters.inpainter import InpainterModule
from typing import Tuple


class MNISTLinearInpainter(InpainterModule):
    """
    Initial inpainter used for MNIST. It has *only* linear layers and it's performance is not reported in the paper.
    """

    def __init__(
        self,
        n_mixes: int = 1,
        a_width: int = 3,
        hidden_size=1024,
        n_hidden_layers: int = 3,
    ):
        super().__init__()

        h = 28
        w = 28
        c = 1
        in_size = c * h * w

        extractor_layers = [
            Reshape((-1, in_size * 2)),
            nn.Linear(in_size * 2, hidden_size),
            nn.ReLU(),
        ]

        for i in range(n_hidden_layers):
            extractor_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        self.extractor = nn.Sequential(*extractor_layers)

        self.a_extractor = nn.Sequential(
            nn.Linear(hidden_size, in_size * n_mixes * a_width),
            Reshape((-1, n_mixes, a_width, in_size)),
        )
        self.m_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size), Reshape((-1, n_mixes, in_size))
        )

        self.d_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size)),
            LambdaLayer(lambda d: torch.sigmoid(d) + 1e-10),
        )

        self.p_extractor = nn.Sequential(nn.Linear(hidden_size, n_mixes), nn.Softmax())

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


class MisganImputer(nn.Module):
    """
    Copied from
    https://github.com/steveli/misgan/blob/master/misgan.ipynb
    """

    def __init__(self, arch=(512, 512)):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(784, arch[0]),
            nn.ReLU(),
            nn.Linear(arch[0], arch[1]),
            nn.ReLU(),
            nn.Linear(arch[1], arch[0]),
            nn.ReLU(),
            nn.Linear(arch[0], 784),
        )

    def forward(self, data, mask, noise):
        net = data * mask + noise * (1 - mask)
        net = net.view(data.shape[0], -1)
        net = self.fc(net)
        net = torch.sigmoid(net).view(data.shape)
        return data * mask + net * (1 - mask), net


class MNISTMisganInpainterInterface(InpainterModule):
    def __init__(self, a_width: int = 3):
        super().__init__(a_width)
        self.imputer = MisganImputer()

    def forward(self, X: torch.Tensor, J: torch.Tensor):
        X_masked = X * J
        batch_size = X.shape[0]
        device = next(self.parameters()).device
        impu_noise = torch.empty(batch_size, 1, 28, 28, device=device)
        impu_noise.uniform_()
        _, m = self.imputer(X_masked, J, impu_noise)
        m = m.reshape(batch_size, 1, -1)
        p = torch.ones(size=(batch_size, 1)).to(device)
        a = torch.zeros(size=(batch_size, 1, self.a_width, 28 * 28))
        d = torch.zeros_like(m)
        return p, m, a, d
