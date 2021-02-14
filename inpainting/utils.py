import json
from pathlib import Path
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from inpainting.datasets.mask_coding import UNKNOWN_LOSS
from inpainting.inpainters.inpainter import InpainterModule


def inpainted(x: np.ndarray, j: np.ndarray, m: np.ndarray):
    x = x.copy()
    x[j == UNKNOWN_LOSS] = m[j == UNKNOWN_LOSS]
    return x


def classifier_experiment(
    inpainter: InpainterModule,
    classifier: MLPClassifier,
    X: np.ndarray,
    J: np.ndarray,
    y: np.ndarray,
):
    y_pred = classifier.predict(X), y
    X_masked = X * J
    y_masked_pred = classifier.predict(X_masked)
    P, M, A, D = inpainter(torch.tensor(X_masked), torch.tensor(J))

    X_inpainted = inpainted(X_masked, J, M[:, 0].detach().cpu().numpy())
    y_inpainted_pred = classifier.predict(X_inpainted)

    return (
        tuple([t.cpu().detach().numpy() for t in [P, M, A, D]]),
        (y_pred, y_masked_pred, y_inpainted_pred),
        X_inpainted,
    )


def predictions_for_entire_loader(
    inpainter: InpainterModule,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu"),
) -> List[Tuple[np.ndarray, ...]]:
    inpainter.to(device)
    results = []
    for i, ((x, j), y) in tqdm(enumerate(data_loader)):
        x, j, y = [t.to(device) for t in [x, j, y]]
        #         print(x.shape)
        p, m, a, d = inpainter(x, j)

        for (x_, j_, y_, p_, m_, a_, d_) in zip(x, j, y, p, m, a, d):
            (x_, j_, y_, p_, m_, a_, d_) = [
                t.detach().cpu().numpy() for t in (x_, j_, y_, p_, m_, a_, d_)
            ]
            results.append((x_, j_, p_, m_, a_, d_, y_))

            # print([t.shape for t in (x_, j_, p_, m_, a_, d_,)])
    return results


# control which parameters are frozen / free for optimization
def free_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def freeze_params(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def printable_history(history: List[Dict]) -> List[Dict]:
    return [
        {k: v for (k, v) in h.items() if k not in ["sample_results"]} for h in history
    ]


def dump_history(history: List[Dict], experiment_path: Path):
    with (experiment_path / "history.json").open("w") as f:
        json.dump(printable_history(history), f, indent=2)

    for metric_name in set(history[0]["metrics"].keys()):
        for fold in ["train", "val"]:
            plt.plot(
                [h["epoch"] for h in history],
                [h["metrics"][metric_name][fold] for h in history],
                label=fold,
            )
        plt.title(metric_name)
        plt.legend()
        fig = plt.gcf()
        fig.savefig(
            experiment_path / f"history.{metric_name}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.show()
        plt.clf()
