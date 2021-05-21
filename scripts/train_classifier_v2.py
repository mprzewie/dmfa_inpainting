#!/usr/bin/env python
# coding: utf-8
import sys

sys.path.append("..")

from pprint import pprint

try:
    import cuml, cudf
except:
    print("Failed to import CUML. This is not breaking unless you want to use KNN.")

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST

import common as common_loaders

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
from inpainting.datasets.cifar import train_val_datasets as cifar_train_val_ds

from inpainting.visualizations.digits import img_with_mask
from inpainting.custom_layers import ConVar, ConVarNaive
from inpainting.inpainters import mocks as inpainters_mocks
from inpainting.classification.training import train_pre_inpainted_classifier
from inpainting.utils import (
    printable_history,
    dump_history,
    predictions_for_entire_loader_as_dataset,
)
from inpainting.classification.inpainting_classifier import (
    PreInpaintedClassifier,
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
    choices=["full", "naive", "partial"],
    help="ConVar implementation to use.",
)

experiment_args.add_argument(
    "--convar_append_mask",
    action="store_true",
    default=False,
    help="If True, known/missing mask will be appended to convar output.",
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
    choices=["gt", "noise", "zero", "dmfa", "mfa", "knn", "acflow"],
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

classifier_args.add_argument(
    "--cls_depth", type=int, default=2, help="Classifier depth"
)

classifier_args.add_argument(
    "--cls_bl", type=int, default=1, help="Classifier conv-relu-bn block len"
)

classifier_args.add_argument(
    "--cls_latent_size", type=int, default=20, help="Classifier latent size"
)

classifier_args.add_argument(
    "--cls_dropout", type=float, default=0.0, help="Classifier dropout"
)

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
experiment_path = (
    args.results_dir / args.dataset / "pre_inpainting" / args.experiment_name
)
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
    print("#", datetime.now(), file=f)
    print("python", *sys.argv, file=f)

img_size = args.img_size


mask_configs_train, mask_configs_val = common_loaders.mask_configs_from_args(args)


if "mnist" in args.dataset:
    ds_train, ds_val = mnist_train_val_ds(
        ds_type=FashionMNIST if args.dataset == "fashion_mnist" else MNIST,
        save_path=Path(args.dataset_root),
        mask_configs_train=mask_configs_train,
        mask_configs_val=mask_configs_val,
        resize_size=(img_size, img_size),
    )
elif args.dataset == "svhn":
    ds_train, ds_val = svhn_train_val_ds(
        save_path=Path(args.dataset_root),
        mask_configs_train=mask_configs_train,
        mask_configs_val=mask_configs_val,
        resize_size=(img_size, img_size),
    )
elif args.dataset == "cifar10":
    ds_train, ds_val = cifar_train_val_ds(
        save_path=Path(args.dataset_root),
        mask_configs_train=mask_configs_train,
        mask_configs_val=mask_configs_val,
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


dl_train_inp = DataLoader(ds_train, args.batch_size, shuffle=False, drop_last=True)
dl_val_inp = DataLoader(ds_val, args.batch_size, shuffle=False, drop_last=True)


img_shape = args.img_size


if args.inpainter_type == "dmfa":
    print(f"Loading DMFA inpainter from {args.inpainter_path}")
    dmfa_path = args.inpainter_path

    with (dmfa_path / "args.json").open("r") as f:
        dmfa_args = dot(json.load(f))

    inpainter = common_loaders.dmfa_from_args(dmfa_args)

    checkpoint = torch.load((dmfa_path / "training.state"), map_location="cpu")
    inpainter.load_state_dict(checkpoint["inpainter"])

elif args.inpainter_type == "mfa":
    print(f"Loading MFA inpainter from {args.inpainter_path}")
    inpainter = common_loaders.mfa_from_path(args.inpainter_path)

elif args.inpainter_type == "acflow":
    inpainter = common_loaders.acflow_from_path(args.inpainter_path, args.batch_size)

elif args.inpainter_type == "gt":
    inpainter = inpainters_mocks.GroundTruthInpainter()

elif args.inpainter_type == "zero":
    inpainter = inpainters_mocks.ZeroInpainter()

elif args.inpainter_type == "noise":
    inpainter = inpainters_mocks.ZeroInpainter()

elif args.inpainter_type == "knn":
    inpainter = inpainters_mocks.KNNInpainter(ds_train)

else:
    raise RuntimeError(f"Unknown inpainter type: {args.inpainter_type}!")


ds_val_inp, ds_train_inp = [
    predictions_for_entire_loader_as_dataset(inpainter, dl, device)
    for dl in [
        tqdm(dl_val_inp, "Inpainting val DS"),
        tqdm(dl_train_inp, "Inpainting train DS"),
    ]
]

dl_train = DataLoader(ds_train_inp, args.batch_size, shuffle=True)

dl_train_val = DataLoader(ds_train_inp, args.batch_size, shuffle=False)
dl_val = DataLoader(ds_val_inp, args.batch_size, shuffle=False)

convar = common_loaders.convar_from_args(args)

classifier = get_classifier(
    in_channels=args.convar_channels,
    in_height=img_shape,
    in_width=img_shape,
    n_classes=args.n_classes,
    depth=args.cls_depth,
    block_len=args.cls_bl,
    latent_size=args.cls_latent_size,
    dropout=args.cls_dropout,
)

pre_inpainted_classifier = PreInpaintedClassifier(convar, classifier).to(device)


optimizer = torch.optim.Adam(pre_inpainted_classifier.parameters(), lr=args.lr)

# save schemas of the inpainter and optimizer
with (experiment_path / "classifier.schema").open("w") as f:
    print(pre_inpainted_classifier, file=f)

with (experiment_path / "opt.schema").open("w") as f:
    print(optimizer, file=f)

history = train_pre_inpainted_classifier(
    classifier=pre_inpainted_classifier,
    data_loader_train=dl_train,
    data_loaders_val={"train": dl_train_val, "val": dl_val},
    optimizer=optimizer,
    n_epochs=args.num_epochs,
    device=device,
    max_benchmark_batches=args.max_benchmark_batches,
)
pprint(printable_history(history))


dump_history(history, experiment_path)

# TODO turn this into functions
if args.dump_sample_results:
    epochs_path = experiment_path / "epochs"
    if epochs_path.exists():
        shutil.rmtree(epochs_path)
    epochs_path.mkdir()

    n_rows_fig = 16

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
                sample_results[k][:n_rows_fig]
                for k in ["X", "J", "P", "M", "A", "D", "Y", "Y_pred"]
            ]

            row_length = vis.row_length(*[t[0] for t in [X, J, P, M, A, D, Y]])

            fig, axes = plt.subplots(n_rows_fig, row_length, figsize=(20, 30))

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
            convar_out = sample_results["convar_out"][:n_rows_fig]

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
                epochs_path / f"{e}_{fold}_convar.png",
                bbox_inches="tight",
                pad_inches=0,
            )
