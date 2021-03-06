import copy
from typing import Callable

import numpy as np
import torch
from torch import nn

from inpainting.datasets import mask_coding as mc


class Reshape(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.out_size)

    def __repr__(self):
        return f"{type(self).__name__}(out_size={self.out_size})"


class LambdaLayer(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor] = lambda x: x):
        super(LambdaLayer, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class ConVar(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, append_mask: bool = False):
        super().__init__()
        self.conv = conv_layer
        self.append_mask = append_mask

    @property
    def conv_squared(self) -> nn.Conv2d:
        conv_sq = copy.deepcopy(self.conv)
        conv_sq.weight = nn.Parameter(self.conv.weight ** 2)
        conv_sq.bias = nn.Parameter(self.conv.bias ** 2)
        return conv_sq

    def forward(self, X, J, P, M, A, D) -> torch.Tensor:
        """
        X: b,c,h,w
        J: b,c,h,w
        P: b,n
        M: b,n,c,h,w
        A: b,n,l,c,h,w
        D: b,n,c,h,w
        """

        b, n, l, c, h, w = A.shape

        oc = self.conv.out_channels

        M_conv_input = M.reshape(b * n, c, h, w)
        D_conv_input = D.reshape(b * n, c, h, w)
        A_conv_input = A.reshape(b * n * l, c, h, w)
        X_conv_input = X * (J == mc.KNOWN)

        if self.append_mask:
            J_md = J.reshape(b, 1, c, h, w).repeat(
                1, n, 1, 1, 1
            )  # b,c,h,w -> b,n,c,h,w
            J_a = J.reshape(b, 1, 1, c, h, w).repeat(
                1, n, l, 1, 1, 1
            )  # b,c,h,w -> b,n,l,c,h,w

            J_md = J_md.reshape(*M_conv_input.shape)
            J_a = J_a.reshape(*A_conv_input.shape)

            M_conv_input = torch.cat([M_conv_input, J_md], dim=1)
            D_conv_input = torch.cat([D_conv_input, J_md], dim=1)
            A_conv_input = torch.cat([A_conv_input, J_a], dim=1)
            X_conv_input = torch.cat([X_conv_input, J], dim=1)

        M_conv = self.conv(M_conv_input).reshape(b, n, oc, h, w)

        # conv squared so that D is positive
        D_conv = self.conv_squared(D_conv_input).reshape(b, n, oc, h, w)
        A_conv = self.conv(A_conv_input).reshape(b, n, l, oc, h, w)

        means = M_conv  # (3)

        variances = D_conv + (A_conv ** 2).sum(
            dim=2
        )  # sum along the L dimension (number of factors)

        stds = variances.sqrt()

        exp_relu_1 = means

        exp_relu_2 = (stds * torch.exp(-(means ** 2) / (2 * variances))) / np.sqrt(
            2 * np.pi
        )

        exp_relu_3 = means * torch.erf(means / (stds * np.sqrt(2)))

        exp_relu = (exp_relu_1 + exp_relu_2 + exp_relu_3) / 2

        exp_relu_weighted_mean = (
            (exp_relu.permute(2, 3, 4, 0, 1) * P).sum(dim=4).permute(3, 0, 1, 2)
        )

        X_known_conv_relu = nn.functional.relu(
            self.conv(X_conv_input)
        )  # known data are simply passed through a convolution

        res = (X_known_conv_relu * (J[:, :1] == mc.KNOWN)) + (
            exp_relu_weighted_mean * (J[:, :1] != mc.KNOWN)
        )

        return res


class ConVarNaive(nn.Module):
    def __init__(self, conv_layer: nn.Conv2d, append_mask: bool = False):
        super().__init__()
        self.conv = conv_layer
        self.append_mask = append_mask

    def forward(self, X, J, P, M, A, D):
        """
        X: b,c,h,w
        J: b,c,h,w
        P: b,n
        M: b,n,c,h,w
        A: b,n,l,c,h,w
        D: b,n,c,h,w
        """

        b, n, l, c, h, w = A.shape

        oc = self.conv.out_channels

        X_inp = (X * (J == mc.KNOWN)) + (M.mean(dim=1) * (J != mc.KNOWN))

        if self.append_mask:
            X_inp = torch.cat([X_inp, J], dim=1)

        res = nn.functional.relu(self.conv(X_inp))

        return res


class PartialConvWrapper(nn.Module):
    def __init__(self, partial_conv: "PartialConv2d"):
        super().__init__()
        self.parconv = partial_conv

    def forward(self, X, J, P, M, A, D):
        # X = (X * (J == mc.KNOWN)) + (M.mean(dim=1) * (J != mc.KNOWN))

        return self.parconv(X, J)
