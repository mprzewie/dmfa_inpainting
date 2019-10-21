import torch
from torch import nn


class GMModel(nn.Module):
    def __init__(
            self, feature_extractor: nn.Module
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.a_extractor = nn.Linear()


    def forward(self, x: torch.Tensor, j: torch.LongTensor):
        features = self.feature_extractor.forward(x, j)

