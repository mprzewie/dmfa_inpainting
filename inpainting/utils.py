import numpy as np
import torch

from inpainting.inpainters.inpainter import InpainterModule
from sklearn.neural_network import MLPClassifier


def inpainted(x: np.ndarray, j: np.ndarray, m: np.ndarray):
    x = x.copy()
    x[j == 0] = m[j == 0]
    return x


def classifier_experiment(
        inpainter: InpainterModule,
        classifier: MLPClassifier,
        X: np.ndarray,
        J: np.ndarray,
        y: np.ndarray
):
    y_pred = classifier.predict(X), y
    X_masked = X * J
    y_masked_pred = classifier.predict(X_masked)
    P, M, A, D = inpainter(torch.tensor(X_masked), torch.tensor(J))

    X_inpainted = inpainted(
        X_masked, J, M[:, 0].detach().cpu().numpy()
    )
    y_inpainted_pred = classifier.predict(X_inpainted)

    return (
        tuple([
            t.cpu().detach().numpy()
            for t in [P, M, A, D]
        ]),
        (
            y_pred, y_masked_pred, y_inpainted_pred
        ),
        X_inpainted
    )
