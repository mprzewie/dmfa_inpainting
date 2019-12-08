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
        for i, ((x,j), y) in dl_iter:
            break
            x, j, y = [t.to(device) for t in [x,j, y]]
            inpainter.zero_grad()
            inpainter.train()
            t1 = time()
            p, m, a, d = inpainter(x, j)
            t2 = time()
            loss = loss_fn(x, j, p, m, a, d)
            t3 = time()
            loss.backward()
            t4 = time()
            optimizer.step()
            t5 = time()
#             print("forward prop", t2 - t1)
#             print("loss calc", t3 - t2)
#             print("backprop", t4 - t3)
#             print("optimizer step", t5 - t4)
#             print("total", t5 - t1)
#             print("----")
            
        inpainter.eval()
        fold_losses = dict()
        sample_results = dict()
        for fold, dl in [
            ("train", data_loader_train),
            ("val", data_loader_val)
        ]:
            losses = []
            for i, ((x,j), y) in enumerate(dl):
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
                if i > 3:
                    break
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