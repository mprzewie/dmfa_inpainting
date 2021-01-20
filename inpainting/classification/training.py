from typing import List, Dict

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from inpainting.classification.inpainting_classifier import InpaintingClassifier
from inpainting.classification.metrics import crossentropy_metric, accuracy_metric


def eval_classifier(
    inpainting_classifier: InpaintingClassifier,
    epoch: int,
    data_loaders: dict,
    device: torch.device,
    metric_fns: dict,
) -> dict:
    inpainting_classifier.eval()
    fold_metrics = dict()
    example_predictions = dict()

    for fold, dl in data_loaders.items():
        metrics = []

        for i, ((X, J), Y) in enumerate(dl):
            X, J, Y = [t.to(device) for t in [X, J, Y]]
            Y_pred, (PMAD_pred, convar_out) = inpainting_classifier(X, J)
            metrics.append(
                {
                    m_name: metric_fn(X, J, Y, Y_pred).item()
                    for m_name, metric_fn in metric_fns.items()
                }
            )
            if i == 0:
                P, M, A, D = PMAD_pred
                preds = dict(
                    X=X,
                    J=J,
                    Y=Y,
                    Y_pred=Y_pred.argmax(dim=1),
                    P=P,
                    M=M,
                    A=A,
                    D=D,
                    convar_out=convar_out,
                )
                example_predictions[fold] = {
                    k: v.cpu().detach().numpy() for (k, v) in preds.items()
                }

        fold_metrics[fold] = metrics

    return dict(
        epoch=epoch,
        metrics={
            m_name: {
                fold: np.mean([m[m_name] for m in f_metrics])
                for fold, f_metrics in fold_metrics.items()
            }
            for m_name in metric_fns.keys()
        },
        sample_results=example_predictions,
    )


def train_classifier(
    inpainting_classifier: InpaintingClassifier,
    data_loader_train: DataLoader,
    data_loaders_val: Dict[str, DataLoader],
    optimizer: Optimizer,
    n_epochs: int,
    device: torch.device,
    tqdm_loader: bool = True,
) -> List[dict]:
    history = []
    epoch = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    history.append(
        eval_classifier(
            inpainting_classifier,
            epoch=epoch,
            data_loaders={
                k: tqdm(v, f"Epoch {epoch}, test_{k}")
                for (k, v) in data_loaders_val.items()
            },
            device=device,
            metric_fns=dict(
                cross_entropy=crossentropy_metric, accuracy=accuracy_metric
            ),
        )
    )
    print(printable_history(history)[-1])

    for e in tqdm(range(1, n_epochs + 1), desc="Epoch"):
        inpainting_classifier.train()

        for (X, J), Y in tqdm(data_loader_train, f"Epoch {e}, train"):
            X, J, Y = [t.to(device) for t in [X, J, Y]]
            Y_pred, _ = inpainting_classifier(X, J)
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        eval_results = eval_classifier(
            inpainting_classifier,
            epoch=e,
            data_loaders={
                k: tqdm(v, f"Epoch {e}, test_{k}")
                for (k, v) in data_loaders_val.items()
            },
            device=device,
            metric_fns=dict(
                cross_entropy=crossentropy_metric, accuracy=accuracy_metric
            ),
        )
        history.append(eval_results)
        print(printable_history([eval_results])[-1])

    return history


def printable_history(history: List[Dict]) -> List[Dict]:
    return [
        {k: v for (k, v) in h.items() if k not in ["sample_results"]} for h in history
    ]
