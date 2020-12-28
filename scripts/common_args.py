from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

import torch

parser = ArgumentParser(
    formatter_class=ArgumentDefaultsHelpFormatter,
    description="Train and benchmark a DMFA inpainter.",
)


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
    default=("cuda:0" if torch.cuda.is_available() else "cpu"),
    help="Torch device to use for computations.",
    type=torch.device,
)


environment_args.add_argument(
    "--dataset_root",
    type=Path,
    default=Path.home() / "uj/.data/",
    help="Path to the dataset files.",
)
