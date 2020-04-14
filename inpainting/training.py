from typing import List, Dict, Optional

import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from inpainting.datasets.mask_coding import KNOWN
from inpainting.inpainters.inpainter import InpainterModule
from inpainting.losses import InpainterLossFn
from torch.cuda import memory_summary
from time import time
from torch.optim.lr_scheduler import _LRScheduler

def num_tensors():
    import torch
    import gc
    res = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                res += 1
#                 print(type(obj), (obj.shape if torch.is_tensor(obj) else obj.data.shape))
        except:
            pass
    print("----")
    return res

def train_step(
    x: torch.Tensor, 
    j: torch.Tensor, 
    inpainter: InpainterModule,
    loss_fn: InpainterLossFn,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
):
    j_unknown = j * (j == KNOWN)
    x_masked = x * j_unknown
    inpainter.zero_grad()
    s = time()
    p, m, a, d = inpainter(x_masked, j_unknown)
    t1 = time()
    loss = loss_fn(x, j, p, m, a, d)
    t2= time()
    loss.backward()
    t3 = time()
    optimizer.step()
    t4 = time()
    if scheduler is not None:
        scheduler.step()
    
#     print([
#         t1 -s, t2 - t1, t3 - t2, t4 - t3
#     ])



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
    scheduler: Optional[_LRScheduler] = None,
) -> List:
    inpainter.train()
    print({
        "all params": sum(p.numel() for p in inpainter.parameters()),
        "backbone params": sum(p.numel() for p in inpainter.extractor.parameters())

    })
    if losses_to_log is None:
        losses_to_log = dict()
    losses_to_log["objective"] = loss_fn
    
    inpainter = inpainter.to(device)

    history = history_start if history_start is not None else [eval_inpainter(
            inpainter,
            epoch=0,
            data_loaders={"train": data_loader_train, "val": data_loader_val},
            device=device,
            losses_to_log=losses_to_log,
            max_benchmark_batches=max_benchmark_batches
        )]

    for e in tqdm(range(n_epochs)):
#         print("num_tensors", num_tensors())

        if tqdm_loader:
            data_loader_train = tqdm(data_loader_train)
        inpainter.train()

        for ((x,j), y) in data_loader_train:
            x, j = [t.to(device) for t in [x,j,]]
            train_step(x, j, inpainter, loss_fn, optimizer, scheduler)


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
            if i > max_benchmark_batches:
                break
            x, j, y = [t.to(device) for t in [x, j, y]]
            p, m, a, d = inpainter(x, j)
            losses.append({
                loss_name: l(x, j, p, m, a, d).detach().cpu().numpy()
                for loss_name, l in losses_to_log.items()
            })
            if i == 0:
                x, j, p, m, a, d, y = [t.detach().cpu().numpy() for t in [x, j, p, m, a, d, y]]
                sample_results[fold] = (
                    x, j, p, m, a, d, y
                )

        fold_losses[fold] = losses
    
#     for k, v in sample_results.items():
#         print([type(t) for t in v])
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