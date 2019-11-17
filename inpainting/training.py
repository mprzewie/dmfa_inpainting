from typing import Tuple, List, Dict

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
    losses_to_log: Dict[str, InpainterLossFn] = None
) -> List:
    if losses_to_log is None:
        losses_to_log = dict()
    losses_to_log["objective"] = loss_fn
    history = []
    for e in tqdm(range(n_epochs)):

        for i, (x_j, y) in enumerate(data_loader_train):
            x = x_j[:, :-1]
            j = x_j[:, -1]
            inpainter.zero_grad()
            inpainter.train()
            p, m, a, d = inpainter(x, j)
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
            for i, (x_j, y) in enumerate(dl):
                x = x_j[:, :-1]
                j = x_j[:, -1]
                p, m, a, d = inpainter(x, j,) # print_features= i ==0 and ((e == (n_epochs -1)) or (e % 15 ==0)))
                losses.append({
                    loss_name: l(x, j, p, m, a, d).detach().cpu().numpy()
                    for loss_name, l in losses_to_log.items()
                })
                if i == 0:
                    x, j, p, m, a, d = [t.detach().cpu().numpy() for t in [x, j, p, m, a, d]]
                    sample_results[fold] = (
                        x, j, p, m, a, d, y
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