#!/usr/bin/env python
# coding: utf-8
import sys

sys.path.append("..")

from pprint import pprint

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST

from common import dmfa_from_args, mfa_from_path

import json
import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

from inpainting.datasets.mask_coding import UNKNOWN_LOSS
from inpainting.datasets.utils import RandomRectangleMaskConfig
from inpainting.datasets.mnist import train_val_datasets as mnist_train_val_ds
from inpainting.datasets.svhn import train_val_datasets as svhn_train_val_ds
from inpainting.visualizations.digits import img_with_mask
from inpainting.custom_layers import ConVar, ConVarNaive
from inpainting.inpainters import mocks as inpainters_mocks
from inpainting.classification.training import train_classifier
from utils import printable_history, dump_history
from inpainting.classification.inpainting_classifier import (
    InpaintingClassifier,
    get_classifier,
)
import inpainting.visualizations.visualizations_utils as vis

from dotted.utils import dot
import numpy as np
import matplotlib
from datetime import datetime

matplotlib.rcParams["figure.facecolor"] = "white"

from common_args import parser, environment_args, experiment_args

experiment_args.add_argument(
    "--convar_type",
    type=str,
    default="full",
    choices=["full", "naive"],
    help="ConVar implementation to use.",
)

experiment_args.add_argument(
    "--seed",
    action="store_true",
    default=False,
    help="If true, random seed will be used for determinism.",
)

experiment_args.add_argument(
    "--train_inpainter_layer",
    action="store_true",
    default=False,
    help="Train inpainter layer with gradient from classification loss. If False, only the classifier layers are trained.",
)


environment_args.add_argument(
    "--results_dir",
    type=Path,
    default="../results/classification",
    help="Base of path where experiment results will be dumped.",
)

inpainter_args = parser.add_argument_group("Inpainter model")

inpainter_args.add_argument(
    "--inpainter_type",
    type=str,
    default="gt",
    choices=["gt", "noise", "zero", "dmfa", "mfa"],
    help="Type of Inpainter. 'gt', 'noise', 'zero' are mock models which produce ground-truth, randn noise and zero imputations, respectively. ",
)

inpainter_args.add_argument(
    "--inpainter_path",
    type=Path,
    help="Used if inpainter_type=dmfa. Should point to DMFA export directory which contains 'args.json' and 'training.state' files.",
    required=False,
    default=None,
)

classifier_args = parser.add_argument_group("Classifier")

classifier_args.add_argument(
    "--convar_channels",
    type=int,
    default=32,
    help="N of output channels of convar layer",
)

classifier_args.add_argument("--n_classes", type=int, default=10, help="N classes")


args = parser.parse_args()

if args.inpainter_type in ["gt", "noise", "zero"]:
    args.inpainter_path = None

if args.seed:
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_deterministic(True)

args_dict = vars(args)
pprint(args_dict)
device = args.device
experiment_path = args.results_dir / args.dataset / args.experiment_name
experiment_path.mkdir(exist_ok=True, parents=True)

print("Experiment path:", experiment_path.absolute())

with (experiment_path / "args.json").open("w") as f:
    json.dump(
        {
            k: v if isinstance(v, (int, str, bool, float)) else str(v)
            for (k, v) in args_dict.items()
        },
        f,
        indent=2,
    )

with (experiment_path / "rerun.sh").open("w") as f:
    print("#", datetime.now())
    print("python", *sys.argv, file=f)

img_size = args.img_size
mask_hidden_h = args.mask_hidden_h
mask_hidden_w = args.mask_hidden_w

if "mnist" in args.dataset:
    ds_train, ds_val = mnist_train_val_ds(
        ds_type=FashionMNIST if args.dataset == "fashion_mnist" else MNIST,
        save_path=Path(args.dataset_root),
        mask_configs=[
            RandomRectangleMaskConfig(UNKNOWN_LOSS, mask_hidden_h, mask_hidden_w)
        ],
        resize_size=(img_size, img_size),
    )
elif args.dataset == "svhn":
    ds_train, ds_val = svhn_train_val_ds(
        save_path=Path(args.dataset_root),
        mask_configs=[
            RandomRectangleMaskConfig(UNKNOWN_LOSS, mask_hidden_h, mask_hidden_w)
        ],
        resize_size=(img_size, img_size),
    )
else:
    raise ValueError(f"Unknown dataset {args.dataset}.")

fig, axes = plt.subplots(4, 4, figsize=(5, 5))
for i in range(16):
    (x, j), y = ds_train[i]
    ax = axes[i // 4, i % 4]
    img_with_mask(x.numpy(), j.numpy(), ax)
    ax.set_title(str(y))
train_fig = plt.gcf()
train_fig.savefig(experiment_path / "train.png")
plt.clf()

print("dataset sizes", len(ds_train), len(ds_val))

dl_train = DataLoader(ds_train, args.batch_size, shuffle=True)
dl_train_val = DataLoader(
    ds_train, 16, shuffle=False
)  # used for validation on training DS
dl_val = DataLoader(ds_val, 16, shuffle=False)

convar_in_channels = 1 if "mnist" in args.dataset else 3
convar = (
    ConVar(
        nn.Conv2d(convar_in_channels, args.convar_channels, kernel_size=3, padding=1)
    )
    if args.convar_type == "full"
    else ConVarNaive(
        nn.Conv2d(convar_in_channels, args.convar_channels, kernel_size=3, padding=1)
    )
)

img_shape = 28 if "mnist" in args.dataset else 32

classifier = get_classifier(
    in_channels=args.convar_channels,
    in_height=img_shape,
    in_width=img_shape,
    n_classes=args.n_classes,
)


if args.inpainter_type == "dmfa":
    print(f"Loading DMFA inpainter from {args.inpainter_path}")
    dmfa_path = args.inpainter_path

    with (dmfa_path / "args.json").open("r") as f:
        dmfa_args = dot(json.load(f))

    inpainter = dmfa_from_args(dmfa_args)

    checkpoint = torch.load((dmfa_path / "training.state"), map_location="cpu")
    inpainter.load_state_dict(checkpoint["inpainter"])

elif args.inpainter_type == "mfa":
    print(f"Loading MFA inpainter from {args.inpainter_path}")
    inpainter = mfa_from_path(args.inpainter_path)

elif args.inpainter_type == "gt":
    inpainter = inpainters_mocks.GroundTruthInpainter()

elif args.inpainter_type == "zero":
    inpainter = inpainters_mocks.ZeroInpainter()

elif args.inpainter_type == "noise":
    inpainter = inpainters_mocks.ZeroInpainter()

else:
    raise RuntimeError(f"Unknown inpainter type: {args.inpainter_type}!")

inpainting_classifier = InpaintingClassifier(
    inpainter,
    convar,
    keep_inpainting_gradient_in_classification=args.train_inpainter_layer,
    classifier=classifier,
).to(device)

optimizer = torch.optim.Adam(inpainting_classifier.parameters(), lr=args.lr)

# save schemas of the inpainter and optimizer
with (experiment_path / "inpainting_classifier.schema").open("w") as f:
    print(inpainting_classifier, file=f)

with (experiment_path / "opt.schema").open("w") as f:
    print(optimizer, file=f)

history = train_classifier(
    inpainting_classifier=inpainting_classifier,
    data_loader_train=dl_train,
    data_loaders_val={"train": dl_train_val, "val": dl_val},
    optimizer=optimizer,
    n_epochs=args.num_epochs,
    device=device,
)
pprint(printable_history(history))


dump_history(history, experiment_path)

# TODO turn this into functions
if args.dump_sample_results:
    epochs_path = experiment_path / "epochs"
    if epochs_path.exists():
        shutil.rmtree(epochs_path)
    epochs_path.mkdir()

    n_rows = 16

    row_length = vis.row_length(
        *[
            history[0]["sample_results"]["train"][t_name][0]
            for t_name in ["X", "J", "P", "M", "A", "D", "Y"]
        ]
    )
    last_epoch = max(h["epoch"] for h in history)

    for h in tqdm(history):
        e = h["epoch"]
        if e % args.render_every != 0 and e != last_epoch:
            continue

        for fold in ["train", "val"]:

            sample_results = h["sample_results"][fold]

            # sample results visualization
            X, J, P, M, A, D, Y, Y_pred = [
                sample_results[k] for k in ["X", "J", "P", "M", "A", "D", "Y", "Y_pred"]
            ]

            row_length = vis.row_length(*[t[0] for t in [X, J, P, M, A, D, Y]])

            fig, axes = plt.subplots(n_rows, row_length, figsize=(20, 30))

            for row_no, (x, j, p, m, a, d, y, y_pred) in enumerate(
                zip(X, J, P, M, A, D, Y, Y_pred)
            ):

                vis.visualize_sample(
                    x,
                    j,
                    p,
                    m,
                    a,
                    d,
                    y,
                    ax_row=axes[row_no],
                    title_prefixes={0: f"{e} {fold} ", 1: f"gt {y} pred {y_pred}"},
                    drawing_fn=img_with_mask,
                )

            title = f"{e}_{fold}_predictions"
            plt.suptitle(title)
            plt.savefig(epochs_path / f"{title}.png", bbox_inches="tight", pad_inches=0)

            # convar out visualization
            convar_out = sample_results["convar_out"][:n_rows]

            n_rows, n_cols = convar_out.shape[:2]

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(
                    n_cols // 2,
                    n_rows // 2,
                ),
            )
            for i, row in enumerate(convar_out):
                for j, img in enumerate(row):
                    ax = axes[i, j]
                    ax.imshow(img)
                    ax.axis("off")

            plt.savefig(
                epochs_path / f"{e}_fold_convar.png", bbox_inches="tight", pad_inches=0
            )
