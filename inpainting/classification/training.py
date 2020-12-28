from typing import List

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from classification.inpainting_classifier import InpaintingClassifier
from classification.metrics import crossentropy_metric, accuracy_metric


def eval_classifier(
    inpainting_classifier: InpaintingClassifier,
    epoch: int,
    data_loaders: dict,
    device: torch.device,
    metric_fns: dict,
) -> dict:
    inpainting_classifier.eval()
    fold_metrics = dict()

    for fold, dl in data_loaders.items():
        metrics = []

        for (X, J), Y in dl:
            X, J, Y = [t.to(device) for t in [X, J, Y]]
            Y_pred, PMAD_pred = inpainting_classifier(X, J)
            metrics.append(
                {
                    m_name: metric_fn(X, J, Y, Y_pred).item()
                    for m_name, metric_fn in metric_fns.items()
                }
            )

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
    )


def train_classifier(
    inpainting_classifier: InpaintingClassifier,
    data_loader_train: DataLoader,
    data_loader_val: DataLoader,
    optimizer: Optimizer,
    n_epochs: int,
    device: torch.device,
    tqdm_loader: bool = True,
) -> List[dict]:
    history = []
    epoch = 0
    loss_fn = torch.nn.CrossEntropyLoss()

    if tqdm_loader:
        data_loader_train = tqdm(data_loader_train, desc="train_loader")
        data_loader_val = tqdm(data_loader_val, desc="val_loader")

    history.append(
        eval_classifier(
            inpainting_classifier,
            epoch=epoch,
            data_loaders={"train": data_loader_train, "val": data_loader_val},
            device=device,
            metric_fns=dict(
                cross_entropy=crossentropy_metric, accuracy=accuracy_metric
            ),
        )
    )

    for e in tqdm(range(1, n_epochs + 1), desc="Epoch"):
        inpainting_classifier.train()

        for (X, J), Y in data_loader_train:
            X, J, Y = [t.to(device) for t in [X, J, Y]]
            Y_pred, PMAD_pred = inpainting_classifier(X, J)
            loss = loss_fn(Y_pred, Y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        eval_results = eval_classifier(
            inpainting_classifier,
            epoch=e,
            data_loaders={"train": data_loader_train, "val": data_loader_val},
            device=device,
            metric_fns=dict(
                cross_entropy=crossentropy_metric, accuracy=accuracy_metric
            ),
        )
        history.append(eval_results)
        print(eval_results)

    return history
