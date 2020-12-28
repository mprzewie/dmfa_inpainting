import torch
from torch import nn
from torchvision.datasets import MNIST, FashionMNIST


device = torch.device(args.device if torch.cuda.is_available() else "cpu")

classifier = nn.Sequential(
    nn
)