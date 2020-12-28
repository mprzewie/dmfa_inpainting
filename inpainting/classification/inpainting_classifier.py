from time import time
from typing import Optional

from torch import nn

from custom_layers import ConVar
from inpainters.inpainter import InpainterModule
from inpainting.datasets.mask_coding import KNOWN


class InpaintingClassifier(nn.Module):
    def __init__(
        self,
        inpainter: InpainterModule,
        convar_layer: ConVar,
        classifier: Optional[nn.Module] = None,
        keep_inpainting_gradient_in_classification: bool = False,
    ):
        super().__init__()
        self.inpainter = inpainter
        self.convar = convar_layer
        self.classifier = classifier or get_classifier(
            in_channels=convar_layer.conv.out_channels
        )
        self.keep_inpainting_gradient_in_classification = (
            keep_inpainting_gradient_in_classification
        )

    def forward(self, X, J):
        t0 = time()
        X_masked = X * (J == KNOWN)
        t1 = time()
        P, M, A, D = self.inpainter(X_masked, J)
        t2 = time()

        b, c, h, w = X.shape
        b, n, chw = M.shape
        b, n, l, chw = A.shape

        P_r = P
        M_r = M.reshape(b, n, c, h, w)
        A_r = A.reshape(b, n, l, c, h, w)
        D_r = D.reshape(b, n, c, h, w)

        if not self.keep_inpainting_gradient_in_classification:
            P_r, M_r, A_r, D_r = [t.detach() for t in [P_r, M_r, A_r, D_r]]

        t3 = time()

        convar_out = self.convar(X, J, P_r, M_r, A_r, D_r)
        t4 = time()
        classification_result = self.classifier(convar_out)
        inpainting_result = (P, M, A, D)
        t5 = time()

        den = t5 - t0
        prep = t1 - t0
        inp = t2 - t1
        resh = t3 - t2
        convar = t4 - t3
        clas = t5 - t4
        times = [prep, inp, resh, convar, clas]

        return classification_result, inpainting_result


def get_classifier(in_channels: int = 32) -> nn.Module:
    """A simple MNIST classifier"""
    return nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(7 * 7 * 64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
