import torch
from torch import nn

from inpainting.custom_layers import Reshape
from inpainting.inpainters.inpainter import InpainterModule


class DigitsLinearInpainter(
    InpainterModule
):
    def __init__(self, n_mixes: int = 1, a_width: int=3, hidden_size: int = 128, n_hidden_layers: int = 2, bias: bool = True, m_sigmoid: bool = False):
        super().__init__(n_mixes=n_mixes, a_width=a_width)
        in_size = 64

        extractor_layers = [
            Reshape((-1, in_size * 2)),
            nn.Linear(in_size * 2, hidden_size, bias=bias),
            nn.ReLU(),
        ]

        for i in range(n_hidden_layers):
            extractor_layers.extend([
                nn.Linear(hidden_size, hidden_size, bias=bias),
                nn.ReLU(),
            ])

        self.extractor = nn.Sequential(*extractor_layers)

        self.a_extractor = nn.Sequential(
            nn.Linear(hidden_size, in_size * n_mixes * a_width, bias=bias),
            Reshape((-1, n_mixes, a_width, in_size,))  # * L, we don't want 1x4 vector but L x4 matrix))
        )
        m_layers = [
            nn.Linear(hidden_size, n_mixes * in_size, bias=bias),
            Reshape((-1, n_mixes, in_size)),
        ]
        if m_sigmoid:
            m_layers.append(nn.Sigmoid())
        self.m_extractor = nn.Sequential(*m_layers)


        self.d_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size, bias=bias),
            Reshape((-1, n_mixes, in_size))

        )

        self.p_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes, bias=bias),
            nn.Softmax(dim=-1)
        )

    def forward(self, X: torch.Tensor, J: torch.Tensor, print_features=False):

        J_unsq = J.unsqueeze(1)
        X_masked = X * J_unsq
        X_J = torch.cat([X_masked, J_unsq], dim=1).float()

        features = self.extractor(X_J)

        p = self.p_extractor(features)
        m = self.m_extractor(features)
        d = self.d_extractor(features)
        a = self.a_extractor(features)

        if print_features:
            print(m)
            m_std = m.std(dim=0)
            print(m_std.min(), m_std.max(), m_std.mean())
            print("----")

        return p, m, a, d