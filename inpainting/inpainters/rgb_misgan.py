# Code adapted from https://github.com/steveli/misgan/tree/master/src
import torch
from torch import nn

from inpainting.inpainters.inpainter import InpainterModule


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        if norm_layer is not None:
            downnorm = norm_layer(inner_nc)
            upnorm = norm_layer(outer_nc)
        uprelu = nn.ReLU(True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv]
            if norm_layer is not None:
                up.append(upnorm)
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv]
            if norm_layer is not None:
                down.append(downnorm)
                up.append(upnorm)

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, layers=5,
                 norm_layer=nn.BatchNorm2d):
        super().__init__()

        mid_layers = layers - 2
        fact = 2 ** mid_layers

        unet_block = UnetSkipConnectionBlock(
            ngf * fact, ngf * fact, input_nc=None, submodule=None,
            norm_layer=norm_layer, innermost=True)

        for _ in range(mid_layers):
            half_fact = fact // 2
            unet_block = UnetSkipConnectionBlock(
                ngf * half_fact, ngf * fact, input_nc=None,
                submodule=unet_block, norm_layer=norm_layer)
            fact = half_fact

        unet_block = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=unet_block,
            outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class Imputer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = lambda x: torch.sigmoid(x)

    def forward(self, input, mask, noise):
        net = input * mask + noise * (1 - mask)
        net = self.imputer_net(net)
        net = self.transform(net)
        # NOT replacing observed part with input data for computing
        # autoencoding loss.
        # return input * mask + net * (1 - mask)
        return net


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
        _, m = self.imputer(X_masked, J_bhw, impu_noise)

        m = m.reshape(b, 1, -1)
        p = torch.ones(size=(b, 1,)).to(device)
        a = torch.zeros(size=(b, 1, self.a_width, -1))
        d = torch.zeros_like(m)
        return p, m, a, d
