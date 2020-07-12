from torch import nn


class InpainterModule(nn.Module):
    def __init__(self, n_mixes: int = 1, a_width: int = 3):
        super().__init__()
        self.n_mixes = n_mixes
        self.a_width = a_width
