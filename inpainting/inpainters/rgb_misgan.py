# Code adapted from https://github.com/steveli/misgan/tree/master/src
import torch
from torch import nn

from inpainting.backbones import UNet
from inpainting.inpainters.inpainter import InpainterModule


class Imputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = lambda x: torch.sigmoid(x)

    def forward(self, input, mask, noise):
        net = input * mask
        net = net + noise * (1- mask)
        net = self.imputer_net(net)
        net = self.transform(net)
        # NOT replacing observed part with input data for computing
        # autoencoding loss.
        # return input * mask + net * (1 - mask)
        return net, net


class UNetImputer(Imputer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.imputer_net = UNet(*args, **kwargs)


class RGBMisganInpainterInterface(
    InpainterModule
):
    def __init__(self, a_width: int = 3):
        super().__init__(a_width)
        self.imputer = UNetImputer()

    def forward(self, X: torch.Tensor, J: torch.Tensor):
        X_masked = X * J
        device = next(self.parameters()).device
        b, c, h, w = X.shape
        impu_noise = torch.empty(b, c, h, w, device=device)
        impu_noise.uniform_()
        
        J_bhw = J.mean(dim=1) # flatten along channels dimension

        _, m = self.imputer(X_masked, J, impu_noise)

        m = m.reshape(b, 1, -1)
        p = torch.ones(size=(b, 1,)).to(device)
        a = torch.zeros(size=(b, 1, self.a_width, c*h*w))
        d = torch.zeros_like(m)
        return p, m, a, d
