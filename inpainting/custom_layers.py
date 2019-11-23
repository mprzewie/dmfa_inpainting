import torch

from torch import nn


class Reshape(nn.Module):
    def __init__(self, out_size):
        super().__init__()

        self.out_size = out_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.out_size)

    def __repr__(self):
        return f"{type(self).__name__}(out_size={self.out_size})"
