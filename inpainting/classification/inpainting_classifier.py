from time import time
from typing import Optional

from torch import nn

from inpainting.custom_layers import ConVar
from inpainting.inpainters.inpainter import InpainterModule
from inpainting import backbones as bkb


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


class PreInpaintedClassifier(nn.Module):
    def __init__(
        self,
        convar_layer: ConVar,
        classifier: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.convar = convar_layer
        self.classifier = classifier or get_classifier(
            in_channels=convar_layer.conv.out_channels
        )

    def forward(self, X, J, P, M, A, D):
        t0 = time()
        b, c, h, w = X.shape
        b, n, chw = M.shape
        b, n, l, chw = A.shape

        P_r = P
        M_r = M.reshape(b, n, c, h, w)
        A_r = A.reshape(b, n, l, c, h, w)
        D_r = D.reshape(b, n, c, h, w)

        t1 = time()

        convar_out = self.convar(X, J, P_r, M_r, A_r, D_r)

        t2 = time()
        classification_result = self.classifier(convar_out)

        inpainting_result = (P, M, A, D)
        t3 = time()

        prep = t1 - t0
        convar = t2 - t1
        clas = t3 - t2
        times = [prep, convar, clas]

        return classification_result, (inpainting_result, convar_out)


def get_classifier(
    in_channels: int = 32,
    in_height: int = 28,
    in_width: int = 28,
    n_classes: int = 10,
    depth: int = 2,
    block_len: int = 1,
    latent_size: int = 20,
    dropout: float = 0.0,
) -> nn.Module:
    """A simple classifier (the first layer is ConVar)"""
    encoder, _ = bkb.down_up_backbone_v2(
        chw=(
            in_channels,
            in_height,
            in_width,
        ),
        depth=depth,
        block_length=block_len,
        first_channels=in_channels,
        latent_size=latent_size,
        dropout=dropout,
    )

    return nn.Sequential(
        encoder, nn.Dropout(p=dropout), nn.ReLU(), nn.Linear(latent_size, n_classes)
    )
