from time import time
from typing import Optional

from torch import nn

from inpainting.custom_layers import ConVar
from inpainting.inpainters.inpainter import InpainterModule


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
        t1 = time()
        P, M, A, D = self.inpainter(X, J)
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
        # classification_result = self.classifier(X)

        inpainting_result = (P, M, A, D)
        t5 = time()

        den = t5 - t0
        prep = t1 - t0
        inp = t2 - t1
        resh = t3 - t2
        convar = t4 - t3
        clas = t5 - t4
        times = [prep, inp, resh, convar, clas]

        return classification_result, (inpainting_result, convar_out)


def get_classifier(
    in_channels: int = 32, in_height: int = 28, in_width: int = 28, n_classes: int = 10
) -> nn.Module:
    """A simple classifier (the first layer is ConVar)"""
    conv_out_chan = 64
    lin_in_shape = (in_height // 4) * (in_width // 4) * conv_out_chan

    return nn.Sequential(
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(in_channels, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(lin_in_shape, 128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    )