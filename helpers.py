import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as T
import multiprocessing
import random
import numpy as np

# import matplotlib.pyplot as plt


def get_data_loaders(batch_size, val_fraction=0.2):
    transforms = T.ToTensor()

    num_workers = multiprocessing.cpu_count()

    # Get Train, Validation, Test datasets

    trainval_data = datasets.MNIST(
        "data", train=True, download=True, transform=transforms
    )

    # Split in train and validation
    train_len = int(len(trainval_data) * (1 - val_fraction))
    val_len = len(trainval_data) - train_len

    print(
        f"Using {train_len} examples for training and \
          {val_len} for validation"
    )

    train_subset, val_subset = random_split(
        trainval_data, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # Get Test Data
    test_data = datasets.MNIST("data", train=False, download=True, transform=transforms)

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"Using {len(test_loader)} examples for testing")

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
