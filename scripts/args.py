from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)


experiment_args = parser.add_argument_group("Experiment")
experiment_args.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Experiment name. Experiment results (model, plots, metrics, etc.) will be saved at: <results_dir>/<dataset>/<experiment_name>",
)
experiment_args.add_argument(
    "--max_benchmark_batches",
    type=int,
    default=200,
    help="Maximal number of batches to process during evaluation",
)

experiment_args.add_argument(
    "--render_every",
    type=int,
    default=5,
    help="Dump inpaintings of model every N epochs. If this value is negative (e.g. -1), rendering is omitted.",
)

experiment_args.add_argument(
    "--dump_val_predictions",
    dest="dump_val_predictions",
    action="store_true",
    default=False,
    help="Whether to dump predictions for the entire validation dataset with the final model after training.",
)
experiment_args.add_argument(
    "--dump_sample_results",
    dest="dump_sample_results",
    action="store_true",
    default=False,
    help="Whether to dump predictions for the entire validation dataset with the final model after training.",
)

data_args = parser.add_argument_group("Dataset")

data_args.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    choices=["mnist", "fashion_mnist", "celeba"],
    help="Dataset to experiment on.",
)
data_args.add_argument(
    "--img_size",
    type=int,
    default=28,
    help="Size to which images from dataset will be resized (both height and width).",
)
data_args.add_argument(
    "--mask_hidden_h",
    type=int,
    default=14,
    help="Height of hidden data masks which will be sampled on top of images.",
)
data_args.add_argument(
    "--mask_hidden_w",
    type=int,
    default=14,
    help="Width of hidden data masks which will be sampled on top of images.",
)

model_args = parser.add_argument_group("Inpainter model")
model_args.add_argument(
    "--architecture",
    type=str,
    default="linear_heads",
    choices=["fullconv", "linear_heads"],
    help="Model architecture to use.",
)

model_args.add_argument(
    "--bkb_fc",
    type=int,
    default=32,
    help="Number of channels in the first convolution of model backbone",
)

model_args.add_argument(
    "--bkb_lc",
    type=int,
    default=32,
    help="Number of channels in the last convolution of model backbone",
)

model_args.add_argument(
    "--bkb_depth",
    type=int,
    default=2,
    help="Number of conv-relu-batchnorm blocks in fully convolutional backbone. Irrelevant when architecture is `linear_heads`.",
)

model_args.add_argument(
    "--bkb_block_length",
    type=int,
    default=1,
    help="Number of repetitions of conv-relu-batchnorm sequence in fully convolutional. Irrelevant when architecture is `linear_heads`.",
)
model_args.add_argument(
    "--bkb_latent",
    dest="bkb_latent",
    action="store_true",
    default=False,
    help="Whether the fully convolutional backbone should have a linear layer between downsampling and upsampling sections. Irrelevant when architecture is `linear_heads`.",
)

model_args.add_argument(
    "--num_factors",
    type=int,
    default=4,
    help="Number of factors / width of matrix A returned by the model, which is used to estimate covariance. In the paper, this value is often referred to as `l`.",
)

model_args.add_argument(
    "--a_amplitude",
    type=float,
    default=0.5,
    help="Amplitude of sigmoid activation through which factor (A) matrices are passed. Amplitude X means that factor matrix can take values in range (-X/2, X/2)",
)

training_args = parser.add_argument_group("Training")

training_args.add_argument(
    "--batch_size", type=int, default=24, help="Batch size during model training."
)

training_args.add_argument("--lr", type=float, default=4e-5, help="Learning rate")

training_args.add_argument(
    "--num_epochs", type=int, default=20, help="Number of training epochs"
)


environment_args = parser.add_argument_group(
    "Environment", description="Runtime-specific arguments"
)

environment_args.add_argument(
    "--device",
    type=str,
    default=("cuda:0" if torch.cuda.is_available() else "cpu"),
    help="Torch device to use for computations.",
)
environment_args.add_argument(
    "--results_dir",
    type=Path,
    default="../results/inpainting",
    help="Base of path where experiment results will be dumped.",
)

environment_args.add_argument(
    "--dataset_root",
    type=Path,
    default=Path.home() / "uj/.data/",
    help="Path to the dataset files.",
)
