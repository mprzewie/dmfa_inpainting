from typing import List, Dict, Optional

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from inpainting.inpainters.inpainter import InpainterModule
from inpainting.losses import InpainterLossFn



def train_inpainter(
    inpainter: InpainterModule,
    data_loader_train: DataLoader,
    data_loader_val: DataLoader,
    optimizer: Optimizer,
    loss_fn: InpainterLossFn,
    n_epochs: int,
    losses_to_log: Dict[str, InpainterLossFn] = None,
    device: torch.device = torch.device("cpu"),
    tqdm_loader: bool = False,
    history_start: Optional[List] = None,
    max_benchmark_batches: int = 50,
) -> List:
    if losses_to_log is None:
        losses_to_log = dict()
    losses_to_log["objective"] = loss_fn

    history = history_start if history_start is not None else [eval_inpainter(
            inpainter,
            epoch=0,
            data_loaders={"train": data_loader_train, "val": data_loader_val},
            device=device,
            losses_to_log=losses_to_log,
            max_benchmark_batches=max_benchmark_batches
        )]

    inpainter = inpainter.to(device)
    for e in tqdm(range(n_epochs)):
        dl_iter = enumerate(data_loader_train)
        if tqdm_loader:
            dl_iter = tqdm(dl_iter)
        for i, ((x,j), y) in dl_iter:
            x, j, y = [t.to(device) for t in [x,j, y]]
            inpainter.zero_grad()
            inpainter.train()
            p, m, a, d = inpainter(x, j)
            loss = loss_fn(x, j, p, m, a, d)
            loss.backward()
            optimizer.step()

        history_elem = eval_inpainter(
            inpainter,
            epoch=e,
            data_loaders={"train": data_loader_train, "val": data_loader_val},
            device=device,
            losses_to_log=losses_to_log,
            max_benchmark_batches=max_benchmark_batches
        )
        history.append(history_elem)

    return history

def eval_inpainter(
    inpainter: InpainterModule,
    epoch: int,
    data_loaders: Dict[str, DataLoader],
    device: torch.device,
    losses_to_log: Dict[str, InpainterLossFn],
    max_benchmark_batches: float,
) -> Dict:
    inpainter.eval()
    fold_losses = dict()
    sample_results = dict()
    for fold, dl in data_loaders.items():
        losses = []
        for i, ((x, j), y) in enumerate(dl):
            x, j, y = [t.to(device) for t in [x, j, y]]
            p, m, a, d = inpainter(x, j)
            losses.append({
                loss_name: l(x, j, p, m, a, d).detach().cpu().numpy()
                for loss_name, l in losses_to_log.items()
            })
            if i == 0:
                x, j, p, m, a, d = [t.detach().cpu().numpy() for t in [x, j, p, m, a, d]]
                sample_results[fold] = (
                    x, j, p, m, a, d, y
                )
            if i < max_benchmark_batches:
                break
        fold_losses[fold] = losses

    return dict(
        epoch=epoch,
        losses={
            loss_name: {
                fold: np.mean([l[loss_name] for l in losses])
                for fold, losses in fold_losses.items()
            }
            for loss_name in losses_to_log.keys()
        },
        sample_results=sample_results
    )