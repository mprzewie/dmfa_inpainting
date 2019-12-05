import torch

from torch import nn
from typing import Callable

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
