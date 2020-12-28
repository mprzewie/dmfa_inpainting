import torch

from torch import nn
from typing import Callable
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
    def __init__(self, conv_layer: nn.Conv2d):
        super().__init__()
        self.conv = conv_layer
        
    
    def forward(
        self, X, J, P, M, A, D
    ):
        """
        X: b,c,h,w
        J: b,c,h,w
        P: b,n
        M: b,n,c,h,w
        A: b,n,l,c,h,w
        D: b,n,c,h,w
        """
        X_masked = X * (J==mc.KNOWN)

        M_proc = ((M.transpose(0,1)*(J!=mc.KNOWN)) + X * (J==mc.KNOWN)).transpose(0,1)
        # m where data is unknown, x where data is known
        
        A_proc = (A.transpose(0,2)*(J!=mc.KNOWN)).transpose(0,2)
        # a where data is unknown, 0 where data is known (no covariance)
        
        D_proc = (D.transpose(0,1)*(J!=mc.KNOWN)).transpose(0,1)
        
        M_proc = (M_proc.permute(2,3,4,0,1) * P).permute(3,4, 0,1,2).sum(dim=1)

        
        return self.conv(M_proc)