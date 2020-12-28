#!/usr/bin/env python
# coding: utf-8
import json
import sys

sys.path.append("..")

from pprint import pprint

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST

from inpainting.classification.inpainting_classifier import InpaintingClassifier
from common import dmfa_from_args


import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

from inpainting.datasets.mask_coding import UNKNOWN_LOSS
from inpainting.datasets.utils import RandomRectangleMaskConfig
from inpainting.datasets.mnist import train_val_datasets as mnist_train_val_ds
from inpainting.visualizations.digits import img_with_mask
from inpainting.custom_layers import ConVar
from inpainting.inpainters import mocks as inpainters_mocks
from inpainting.classification.training import train_classifier
import matplotlib

matplotlib.rcParams["figure.facecolor"] = "white"

from common_args import parser, environment_args

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
    choices=["gt", "noise", "zero"],  # TODO DMFA, MFA
    help="Type of Inpainter. 'gt', 'noise', 'zero' are mock models which produce ground-truth, randn noise and zero imputations, respectively. ",
    # TODO DMFA
    # "If one of those models is chosen, other inpainter parameters are ignored. "
    # "If DMFA is chosen, it must be loaded from a checkpoint."
)

# TODO load DMFA from a directory - parameters are stored in a JSON and checkpoint is stored in the checkpoint.pth
# inpainter_args.add_argument(
#     "--inpainter_checkpoint",
#     type=Path,
#     help="Path to the file with inpainter's checkpoint. Will typically be named 'training.state'.",
#     required=False
# )

args = parser.parse_args()

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
else:
    raise ValueError(
        f"Unknown dataset {args.dataset}. Only MNIST-like datasets supported!"
    )

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
dl_val = DataLoader(ds_val, 16, shuffle=False)

if args.inpainter_type == "dmfa":
    inpainter = dmfa_from_args(args)
    checkpoint = torch.load(args.inpainter_checkpoint, map_location="cpu")
    inpainter.load_state_dict(checkpoint["inpainter"])

elif args.inpainter_type == "gt":
    inpainter = inpainters_mocks.GroundTruthInpainter(a_width=args.num_factors)

elif args.inpainter_type == "zero":
    inpainter = inpainters_mocks.ZeroInpainter(a_width=args.num_factors)

elif args.inpainter_type == "noise":
    inpainter = inpainters_mocks.ZeroInpainter(a_width=args.num_factors)

else:
    raise RuntimeError(f"Unknown inpainter type: {args.inpainter_type}!")


convar = ConVar(nn.Conv2d(1, 32, kernel_size=3, padding=1))

inpainting_classifier = InpaintingClassifier(
    inpainter, convar, keep_inpainting_gradient_in_classification=False
).to(device)

optimizer = torch.optim.Adam(inpainting_classifier.parameters(), lr=args.lr)

history = train_classifier(
    inpainting_classifier=inpainting_classifier,
    data_loader_train=dl_train,
    data_loader_val=dl_val,
    optimizer=optimizer,
    n_epochs=args.num_epochs,
    device=device,
    tqdm_loader=True,
)
# save schemas of the inpainter and optimizer
with (experiment_path / "inpainting_classifier.schema").open("w") as f:
    print(inpainting_classifier, file=f)

with (experiment_path / "opt.schema").open("w") as f:
    print(optimizer, file=f)

pprint(history)

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
    fig.savefig(experiment_path / f"history.{metric_name}.png")
    plt.show()
    plt.clf()
