from collections import Callable

import torch
from torch import nn

ClassificationMetricFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]
"""X, J, Y, Y_pred -> metric"""

crossentropy_metric = lambda X, J, Y, Y_pred: nn.functional.cross_entropy(Y_pred, Y)
accuracy_metric = lambda X, J, Y, Y_pred: (Y_pred.argmax(dim=1) == Y).float().mean()
