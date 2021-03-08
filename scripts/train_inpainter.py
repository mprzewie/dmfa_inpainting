#!/usr/bin/env python
# coding: utf-8

import sys

sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch import optim

from inpainting.losses import nll_masked_batch_loss_components
from pathlib import Path
import inpainting.visualizations.visualizations_utils as vis
import pickle

from pprint import pprint
import json
from tqdm import tqdm
from inpainting.visualizations.stats import plot_arrays_stats
from inpainting.datasets.celeba import train_val_datasets as celeba_train_val_ds
from inpainting.datasets.mnist import train_val_datasets as mnist_train_val_ds
from inpainting.datasets.svhn import train_val_datasets as svhn_train_val_ds
from inpainting.datasets.cifar import train_val_datasets as cifar_train_val_ds

from inpainting.datasets.mask_coding import UNKNOWN_LOSS
from inpainting.datasets.utils import RandomRectangleMaskConfig
from inpainting.visualizations.digits import img_with_mask
from inpainting.training import train_inpainter
from inpainting.utils import predictions_for_entire_loader
from inpainting import losses2 as l2
from torchvision.datasets import MNIST, FashionMNIST
import matplotlib

matplotlib.rcParams["figure.facecolor"] = "white"

from common_args import parser, training_args, environment_args
from common import dmfa_from_args
from datetime import datetime
from inpainting.utils import printable_history, dump_history

training_args.add_argument(
    "--l_nll_weight", type=float, default=1, help="Weight of NLL in the training loss."
)
training_args.add_argument(
    "--l_mse_weight", type=float, default=0, help="Weight of MSE in the training loss."
)

environment_args.add_argument(
    "--results_dir",
    type=Path,
    default="../results/inpainting",
    help="Base of path where experiment results will be dumped.",
)

dmfa_args = parser.add_argument_group("DMFA parameters")
dmfa_args.add_argument(
    "--architecture",
    type=str,
    default="linear_heads",
    choices=["fullconv", "linear_heads"],
    help="DMFA architecture to use.",
)

dmfa_args.add_argument(
    "--bkb_fc",
    type=int,
    default=32,
    help="Number of channels in the first convolution of model backbone",
)

dmfa_args.add_argument(
    "--bkb_lc",
    type=int,
    default=32,
    help="Number of channels in the last convolution of model backbone",
)

dmfa_args.add_argument(
    "--bkb_depth",
    type=int,
    default=2,
    help="Number of conv-relu-batchnorm blocks in fully convolutional backbone. Irrelevant when architecture is `linear_heads`.",
)

dmfa_args.add_argument(
    "--bkb_block_length",
    type=int,
    default=1,
    help="Number of repetitions of conv-relu-batchnorm sequence in fully convolutional. Irrelevant when architecture is `linear_heads`.",
)
dmfa_args.add_argument(
    "--bkb_latent",
    dest="bkb_latent",
    action="store_true",
    default=False,
    help="Whether the fully convolutional backbone should have a linear layer between downsampling and upsampling sections. Irrelevant when architecture is `linear_heads`.",
)

dmfa_args.add_argument(
    "--num_factors",
    type=int,
    default=4,
    help="Number of factors / width of matrix A returned by the model, which is used to estimate covariance. In the paper, this value is often referred to as `l`.",
)

dmfa_args.add_argument(
    "--num_mixes",
    type=int,
    default=1,
    help="Number of GMM mixes.",
)

dmfa_args.add_argument(
    "--a_amplitude",
    type=float,
    default=0.5,
    help="Amplitude of sigmoid activation through which factor (A) matrices are passed. Amplitude X means that factor matrix can take values in range (-X/2, X/2)",
)


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

with (experiment_path / "rerun.sh").open("a") as f:
    print("#", datetime.now(), file=f)
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
elif args.dataset == "celeba":
    img_to_crop = 1.875
    # celebA images are cropped to contain only the faces, which are assumed to be in images' centers
    full_img_size = int(img_size * img_to_crop)
    ds_train, ds_val = celeba_train_val_ds(
        save_path=Path(args.dataset_root),
        mask_configs=[
            RandomRectangleMaskConfig(UNKNOWN_LOSS, mask_hidden_h, mask_hidden_w)
        ],
        resize_size=(full_img_size, full_img_size),
        crop_size=(img_size, img_size),
    )
elif args.dataset == "svhn":
    ds_train, ds_val = svhn_train_val_ds(
        save_path=Path(args.dataset_root),
        mask_configs=[
            RandomRectangleMaskConfig(UNKNOWN_LOSS, mask_hidden_h, mask_hidden_w)
        ],
        resize_size=(img_size, img_size),
    )
elif args.dataset == "cifar10":
    ds_train, ds_val = cifar_train_val_ds(
        save_path=Path(args.dataset_root),
        mask_configs=[
            RandomRectangleMaskConfig(UNKNOWN_LOSS, mask_hidden_h, mask_hidden_w)
        ],
        resize_size=(img_size, img_size),
    )
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

# plot example images from the train dataset

fig, axes = plt.subplots(10, 10, figsize=(15, 15))
for i in range(100):
    (x, j), y = ds_train[i]
    ax = axes[i // 10, i % 10]
    img_with_mask(x.numpy(), j.numpy(), ax)
train_fig = plt.gcf()
train_fig.savefig(experiment_path / "train.png")
plt.clf()

print("dataset sizes", len(ds_train), len(ds_val))

batch_size = args.batch_size
dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_val = DataLoader(ds_val, 16, shuffle=False)

# measure various components of the NLL loss:
log_nominators = lambda x, j, p, m, a, d: nll_masked_batch_loss_components(
    x, j, p, m, a, d
)["log_noms"]

log_determinants = lambda x, j, p, m, a, d: nll_masked_batch_loss_components(
    x, j, p, m, a, d
)["log_dets"]

a_variance = lambda x, j, p, m, a, d: a.var()

# instantiate the inpainter model
inpainter = dmfa_from_args(args)
# print(inpainter)

inpainter.train()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

inpainter = inpainter.to(device)
optimizer = optim.Adam(inpainter.parameters(), lr=args.lr)

# train inpainter and save the history of the training
history = train_inpainter(
    inpainter,
    dl_train,
    dl_val,
    optimizer,
    loss_fn=l2.nll_plus_mse_weighted_loss(args.l_nll_weight, args.l_mse_weight),
    n_epochs=args.num_epochs,
    device=device,
    max_benchmark_batches=args.max_benchmark_batches,
    losses_to_log=dict(
        mse=l2.loss_factory(
            gathering_fn=l2.buffered_gather_batch_by_mask_indices, calc_fn=l2.mse
        ),
        nll=l2.nll_buffered,
        log_nominators=log_nominators,
        log_determinants=log_determinants,
        a_variance=a_variance,
    ),
    tqdm_loader=True,
    export_path=experiment_path,
    export_freq=args.render_every,
    dump_sample_results=args.dump_sample_results,
)
dump_history(history, experiment_path)


# save schemas of the inpainter and optimizer
with (experiment_path / "inpainter.schema").open("w") as f:
    print(inpainter, file=f)
with (experiment_path / "opt.schema").open("w") as f:
    print(optimizer, file=f)

# plot losses / metrics

for loss_name in set(history[0]["metrics"].keys()):
    for fold in ["train", "val"]:
        plt.plot(
            [h["epoch"] for h in history],
            [h["metrics"][loss_name][fold] for h in history],
            label=fold,
        )
    plt.title(loss_name)
    plt.legend()
    fig = plt.gcf()
    fig.savefig(experiment_path / f"history.{loss_name}.png")
    plt.show()
    plt.clf()

# plot example imputations from the history
render_every = args.render_every

if render_every >= 0:

    row_length = vis.row_length(*list(zip(*history[0]["sample_results"]["train"]))[0])
    last_epoch = max(h["epoch"] for h in history)

    fig, axes = plt.subplots(
        len(
            [
                h
                for h in history
                if (h["epoch"] % render_every) == 0 or h["epoch"] == last_epoch
            ]
        )
        * 2,
        row_length,
        figsize=(20, 30),
    )

    row_no = 0
    for h in tqdm(history):

        e = h["epoch"]
        if e % render_every != 0 and e != last_epoch:
            continue

        for ax_no, fold in [(0, "train"), (1, "val")]:
            x, j, p, m, a, d, y = [t[0] for t in h["sample_results"][fold]]
            # print(e, row_no, axes.shape)
            vis.visualize_sample(
                x,
                j,
                p,
                m,
                a,
                d,
                y,
                ax_row=axes[row_no],
                title_prefixes={0: f"{e} {fold} "},
                drawing_fn=img_with_mask,
            )
            row_no += 1

    epochs_fig = plt.gcf()
    epochs_fig.savefig(experiment_path / "epochs_renders.png")

    epochs_path = experiment_path / "epochs"
    epochs_path.mkdir(exist_ok=True)
    n_rows = 16

    for h in tqdm(history):
        e = h["epoch"]
        if e % render_every != 0 and e != last_epoch:
            continue

        for ax_no, fold in [(0, "train"), (1, "val")]:

            row_length = vis.row_length(*list(zip(*h["sample_results"][fold]))[0])

            fig, axes = plt.subplots(n_rows, row_length, figsize=(20, 30))

            for row_no, (x, j, p, m, a, d, y) in enumerate(
                list(zip(*h["sample_results"][fold]))[:n_rows]
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
                    title_prefixes={0: f"{e} {fold} "},
                    drawing_fn=img_with_mask,
                )

            title = f"{e}_{fold}"
            plt.suptitle(title)
            plt.savefig(epochs_path / f"{title}.png")

hist_last_epoch = history[-1]

# analyze model outputs in the final epoch
for fold in ["val"]:
    x, j, p, m, a, d, y = hist_last_epoch["sample_results"][fold]

    a_resh = a.reshape(a.shape[0] * a.shape[1], a.shape[2], a.shape[3])
    covs = a_resh.transpose(0, 2, 1) @ a_resh

    fig, ax = plt.subplots(figsize=(20, 25), nrows=5)

    ax[0].set_title(f"m stats {fold}")
    plot_arrays_stats(m, ax[0])

    samples = [
        vis.gans_gmms_sample_no_d(x_, m_[0], a_[0], d_[0])
        for (x_, m_, a_, d_) in zip(x, m, a, d)
    ]

    ax[1].set_title("samples stats")
    plot_arrays_stats(samples, ax[1])

    ax[2].set_title(f"a stats {fold}")
    plot_arrays_stats(a, ax[2])

    ax[3].set_title(f"d stats {fold}")
    plot_arrays_stats(d, ax[3])

    ax[4].set_title(f"cov stats {fold}")
    plot_arrays_stats(covs, ax[4])
    [a.legend() for a in ax[:5]]
    fig.savefig(experiment_path / f"outputs_stats.png")
    plt.show()

    cov_resh = covs[0].reshape(-1)
    plt.hist(cov_resh, log=True, bins=100)
    plt.title(f"cov[0] hist {fold}")
    plt.show()

    cov = covs[0]

    fig, ax = plt.subplots(figsize=(10, 10), nrows=2)
    eigs = np.linalg.eigvals(cov)
    ax[0].scatter(range(len(eigs)), eigs)
    ax[0].set_title("eigenvals of cov[0]")

    cov_d = cov + np.diag(d[0])
    eigs_d = np.linalg.eigvals(cov_d)
    ax[1].scatter(range(len(eigs_d)), eigs_d)
    ax[1].set_title("eigenvals of cov[0] + d[0]")
    fig.savefig(experiment_path / "eigenvals.png")
    plt.show()

    print("m analysis")

    plt.hist(d[0].reshape(-1), bins=100, log=True)
    plt.title("d[0] hist")
    plt.show()

# dump predictions of the final model for the entire validation dataset
if args.dump_val_predictions:
    val_results = predictions_for_entire_loader(
        inpainter.to(torch.device("cpu")), dl_val, torch.device("cpu")
    )
    with (
        experiment_path / f"val_predictions_{mask_hidden_h}x{mask_hidden_w}.pkl"
    ).open("wb") as f:
        pickle.dump(val_results, f)
