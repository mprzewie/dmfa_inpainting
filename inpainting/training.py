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
import pickle
from pathlib import Path
from inpainting.datasets.mask_coding import UNKNOWN_LOSS, UNKNOWN_NO_LOSS
from inpainting.datasets.utils import RandomRectangleMaskConfig


def num_tensors():
    import torch
    import gc

    res = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
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
    loss = loss_fn(x, j, p.log(), m, a, d)
    t2 = time()
    loss.backward()
    t3 = time()
    optimizer.step()
    t4 = time()
    if scheduler is not None:
        scheduler.step()
    return loss.item()


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
    max_benchmark_batches: int = 50,
    scheduler: Optional[_LRScheduler] = None,
    export_path: Optional[Path] = None,
    export_freq: int = 5,
    dump_sample_results: bool = False,
) -> List:
    print("training on device", device)
    epoch = 0
    history = []

    if export_path is not None:
        export_path.mkdir(parents=True, exist_ok=True)
        state_path = export_path / "training.state"
        if state_path.exists():
            chckp = torch.load(state_path, map_location=device)
            inpainter.load_state_dict(chckp["inpainter"])
            optimizer.load_state_dict(chckp["optimizer"])
            epoch = chckp["epoch"]

        histories_paths = (export_path / "histories").glob("*.pkl")
        for hp in histories_paths:
            with hp.open("rb") as f:
                history.append(pickle.load(f))
        history = sorted(history, key=lambda h: h["epoch"])
        print("Loaded history from epochs", [h["epoch"] for h in history])

    print(f"starting training from epoch {epoch}")

    inpainter.train()
    print(
        {
            "all params": sum(p.numel() for p in inpainter.parameters()),
            "backbone params": sum(p.numel() for p in inpainter.extractor.parameters()),
        }
    )
    if losses_to_log is None:
        losses_to_log = dict()
    losses_to_log["objective"] = loss_fn

    inpainter = inpainter.to(device)

    history = (
        history
        if len(history) > 0
        else [
            eval_inpainter(
                inpainter,
                epoch=0,
                data_loaders={"train": data_loader_train, "val": data_loader_val},
                device=device,
                losses_to_log=losses_to_log,
                max_benchmark_batches=max_benchmark_batches,
            )
        ]
    )

    for e in tqdm(range(epoch, n_epochs + epoch), desc="Epoch"):
        inpainter.train()

        for ((x, j), y) in tqdm(data_loader_train, desc=f"Epoch {e}, train"):
            x, j = [t.to(device) for t in [x, j]]
            loss = train_step(x, j, inpainter, loss_fn, optimizer, scheduler)
            if np.isnan(loss):
                print(f"stoping at epoch {e} bc loss is nan")
                return history
        history_elem = eval_inpainter(
            inpainter,
            epoch=e,
            data_loaders={
                "train": tqdm(data_loader_train, desc=f"Epoch {e}, test_train"),
                "val": tqdm(data_loader_val, desc=f"Epoch {e}, test_val"),
            },
            device=device,
            losses_to_log=losses_to_log,
            max_benchmark_batches=max_benchmark_batches,
        )
        history.append(history_elem)

        if export_path is not None and (
            (e % export_freq) == 0 or e == (epoch + n_epochs - 1)
        ):

            if dump_sample_results:
                histories_path = export_path / "histories"
                histories_path.mkdir(exist_ok=True)
                with (histories_path / f"{e}.pkl").open("wb") as f:
                    pickle.dump(history_elem, f)

            state_dict = {
                "inpainter": inpainter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": e,
            }

            state_file = export_path / "training.state"

            if state_path.exists():
                state_path.unlink()

            torch.save(state_dict, state_path)

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
        for i, ((x, j), y) in tqdm(enumerate(dl), f"Epoch {epoch} - {fold}"):
            if i > max_benchmark_batches:
                break
            x, j, y = [t.to(device) for t in [x, j, y]]
            p, m, a, d = inpainter(x, j)
            losses.append(
                {
                    loss_name: l(x, j, p.log(), m, a, d).detach().item()
                    for loss_name, l in losses_to_log.items()
                }
            )
            if i == 0:
                x, j, p, m, a, d, y = [
                    t.detach().cpu().numpy() for t in [x, j, p, m, a, d, y]
                ]
                sample_results[fold] = (x, j, p, m, a, d, y)

        fold_losses[fold] = losses

    return dict(
        epoch=epoch,
        metrics={
            loss_name: {
                fold: np.mean([l[loss_name] for l in losses])
                for fold, losses in fold_losses.items()
            }
            for loss_name in losses_to_log.keys()
        },
        sample_results=sample_results,
    )
