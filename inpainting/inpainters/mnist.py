import torch
from torch import nn

from inpainting.custom_layers import Reshape, LambdaLayer
from inpainting.datasets.mask_coding import KNOWN
from inpainting.inpainters.inpainter import InpainterModule


class MNISTLinearInpainter(
    InpainterModule
):
    def __init__(self, n_mixes: int = 1, a_width: int=3, hidden_size=1024, n_hidden_layers: int = 3):
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
            extractor_layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ])

        self.extractor = nn.Sequential(*extractor_layers)

        self.a_extractor = nn.Sequential(
            nn.Linear(hidden_size, in_size * n_mixes * a_width),
            Reshape((-1, n_mixes, a_width, in_size,))
        )
        self.m_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size)),

        )

        self.d_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes * in_size),
            Reshape((-1, n_mixes, in_size)),
            LambdaLayer(lambda d: torch.sigmoid(d) + 1e-10),
        )

        self.p_extractor = nn.Sequential(
            nn.Linear(hidden_size, n_mixes),
            nn.Softmax()
        )

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
    

    import torch
from torch import nn

from inpainting.custom_layers import Reshape, LambdaLayer
from inpainting.inpainters.inpainter import InpainterModule


class MNISTConvolutionalInpainter(
    InpainterModule
):
    def __init__(self, n_mixes: int = 1, a_width: int=3):
        super().__init__()

        h = 28
        w = 28
        c = 1
        in_size = c * h * w
        last_channels = 128
        hidden_size = h * w * last_channels

        self.extractor = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, padding=2),
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
        
    @staticmethod
    def postprocess_d(d_tensor):
        return torch.sigmoid(d_tensor) + 1e-10
    
    @staticmethod
    def postprocess_a(a_tensor):
        ampl = 0.5
        return ampl * torch.sigmoid(a_tensor) - (ampl / 2)

    def forward(self, X, J):
        X_masked = X * J
        X_J = torch.cat([X_masked, J], dim=1)

        features = self.extractor(X_J)

        m = self.m_extractor(features)
        d = self.d_extractor(features)
        p = self.p_extractor(features)
        a = self.a_extractor(features)

        return p, m, a, d