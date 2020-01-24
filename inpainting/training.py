from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from inpainting.inpainters.inpainter import InpainterModule
from inpainting.losses import InpainterLossFn
from time import time


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
        history_start: Optional[List] = None
) -> List:
    if losses_to_log is None:
        losses_to_log = dict()
    losses_to_log["objective"] = loss_fn
    history = history_start if history_start is not None else []
    inpainter = inpainter.to(device)
    for e in tqdm(range(n_epochs)):
        dl_iter = enumerate(data_loader_train)
        if tqdm_loader:
            dl_iter = tqdm(dl_iter)
        for i, ((x, j, j_2), y) in dl_iter:
            x, j, j_2, y = [t.to(device) for t in [x, j, j_2, y]]
            inpainter.zero_grad()
            inpainter.train()
            p, m, a, d = inpainter(x, j, j_2)
            loss = loss_fn(x, j, p, m, a, d)
            loss.backward()
            optimizer.step()

        inpainter.eval()
        fold_losses = dict()
        sample_results = dict()
        for fold, dl in [
            ("train", data_loader_train),
            ("val", data_loader_val)
        ]:
            losses = []
            for i, ((x, j, j_2), y) in enumerate(dl):
                x, j, j_2, y = [t.to(device) for t in [x, j, j_2, y]]
                p, m, a, d = inpainter(x, j, j_2)

                # loss is calculated only from j, i.e. the mask under which we know what is behind
                losses.append({
                    loss_name: l(x, j, p, m, a, d).detach().cpu().numpy()
                    for loss_name, l in losses_to_log.items()
                })
                if i == 0:
                    x, j, j_2, p, m, a, d = [t.detach().cpu().numpy() for t in [x, j, j_2, p, m, a, d]]
                    sample_results[fold] = (
                        x, j, j_2, p, m, a, d, y
                    )
            fold_losses[fold] = losses

        history.append(dict(
            losses={
                loss_name: {
                    fold: np.mean([l[loss_name] for l in losses])
                    for fold, losses in fold_losses.items()
                }
                for loss_name in losses_to_log.keys()
            },
            sample_results=sample_results
        ))

    return history
