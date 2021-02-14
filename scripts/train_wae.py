#!/usr/bin/env python
# coding: utf-8
import sys

from generative.wae import InpaintingWAE, get_discriminator, train_wae

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
from utils import printable_history, dump_history
import inpainting.visualizations.visualizations_utils as vis
import inpainting.backbones as bkb
from dotted.utils import dot
import numpy as np

import matplotlib

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
    default="../results/generation",
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

wae_args = parser.add_argument_group("WAE model")

wae_args.add_argument(
    "--wae_depth",
    type=int,
    help="Number of conv-relu-batchnorm blocks in fully convolutional WAE",
)

wae_args.add_argument(
    "--wae_bl",
    type=int,
    default=1,
    help="Number of repetitions of conv-relu-batchnorm sequence in fully convolutional WAE.",
)

wae_args.add_argument(
    "--wae_fc",
    type=int,
    default=32,
    help="Number of channels in the first convolution of WAE encoder",
)

wae_args.add_argument(
    "--wae_lc",
    type=int,
    default=32,
    help="Number of channels in the last convolution of WAE decoder",
)

wae_args.add_argument(
    "--wae_disc_hidden", type=int, default=64, help="Hidden size of WAE discriminator",
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
convar_out_channels = args.convar_channels
convar = (
    ConVar(nn.Conv2d(convar_in_channels, convar_out_channels, kernel_size=3, padding=1))
    if args.convar_type == "full"
    else ConVarNaive(
        nn.Conv2d(convar_in_channels, convar_out_channels, kernel_size=3, padding=1)
    )
)

img_shape = 28 if "mnist" in args.dataset else 32

enc_down, enc_latent, dec = bkb.down_up_backbone(
    chw=(convar_out_channels, img_shape, img_shape),
    depth=args.wae_depth,
    block_length=args.wae_bl,
    first_channels=args.wae_fc,
    last_channels=args.wae_lc,
    latent=True,
)

dec_final_conv = nn.Conv2d(
    args.wae_lc, out_channels=convar_in_channels, kernel_size=3, padding=1
)
wae_encoder = nn.Sequential(enc_down, enc_latent)
wae_decoder = nn.Sequential(dec, dec_final_conv, nn.Sigmoid())

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
    raise TypeError(f"Unknown inpainter {args.inpainter_type}")


discriminator_in_size = (args.wae_lc * (2 ** (args.wae_depth - 1))) * (
    (img_size // (2 ** args.wae_depth)) ** 2
)

wae = InpaintingWAE(
    inpainter=inpainter,
    convar_layer=convar,
    encoder=wae_encoder,
    decoder=wae_decoder,
    discriminator=get_discriminator(
        in_size=discriminator_in_size, hidden_size=args.wae_disc_hidden
    ),
)

optimizer = torch.optim.Adam(wae.parameters(), lr=args.lr)
# save schemas of the inpainter and optimizer
with (experiment_path / "inpainting_wae.schema").open("w") as f:
    print(wae, file=f)

with (experiment_path / "opt.schema").open("w") as f:
    print(optimizer, file=f)


history = train_wae(
    wae=wae,
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

    last_epoch = max(h["epoch"] for h in history)

    for h in tqdm(history):
        e = h["epoch"]
        if e % args.render_every != 0 and e != last_epoch:
            continue

        for fold in ["train", "val"]:

            sample_results = h["sample_results"][fold]

            # sample results visualization
            X, J, P, M, A, D, Y, decoder_out = [
                sample_results[k]
                for k in ["X", "J", "P", "M", "A", "D", "Y", "decoder_out"]
            ]

            row_length = vis.row_length(*[t[0] for t in [X, J, P, M, A, D, Y]]) + 1

            fig, axes = plt.subplots(n_rows, row_length, figsize=(20, 30))

            for row_no, (x, j, p, m, a, d, y, dec_out) in enumerate(
                zip(X, J, P, M, A, D, Y, decoder_out)
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
                    title_prefixes={0: f"{e} {fold} ", 1: f"gt {y}"},
                    drawing_fn=img_with_mask,
                )
                axes[row_no][-1].imshow(dec_out)

            title = f"{e}_{fold}_predictions"
            plt.suptitle(title)
            plt.savefig(epochs_path / f"{title}.png", bbox_inches="tight", pad_inches=0)