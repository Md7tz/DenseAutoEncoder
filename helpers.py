import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision.transforms as T
import multiprocessing
import random
import numpy as np
import matplotlib.pyplot as plt


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

    print(f"Using {train_len} examples for training and {val_len} for validation")

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


def anomaly_detection_display(df):
    df.sort_values(by="loss", ascending=False, inplace=True)

    fig, sub = plt.subplots()
    df["loss"].hist(bins=100)
    sub.set_yscale("log")
    sub.set_xlabel("Score (loss)")
    sub.set_ylabel("Counts per bin")
    fig.suptitle("Distribution of score (loss)")

    fig, subs = plt.subplots(2, 20, figsize=(20, 3))

    for img, sub in zip(df["image"].iloc[:20], subs[0].flatten()):
        sub.imshow(img[0, ...], cmap="gray")
        sub.axis("off")

    for rec, sub in zip(df["reconstructed"].iloc[:20], subs[1].flatten()):
        sub.imshow(rec[0, ...], cmap="gray")
        sub.axis("off")

    fig.suptitle("Most difficult to reconstruct")
    subs[0][0].axis("on")
    subs[0][0].set_xticks([])
    subs[0][0].set_yticks([])
    subs[0][0].set_ylabel("Input")

    subs[1][0].axis("on")
    subs[1][0].set_xticks([])
    subs[1][0].set_yticks([])
    _ = subs[1][0].set_ylabel("Reconst")

    fig, subs = plt.subplots(2, 20, figsize=(20, 3))

    sample = df.iloc[7000:].sample(20)

    for img, sub in zip(sample["image"], subs[0].flatten()):
        sub.imshow(img[0, ...], cmap="gray")
        sub.axis("off")

    for rec, sub in zip(sample["reconstructed"], subs[1].flatten()):
        sub.imshow(rec[0, ...], cmap="gray")
        sub.axis("off")

    fig.suptitle("Sample of in-distribution numbers")
    subs[0][0].axis("on")
    subs[0][0].set_xticks([])
    subs[0][0].set_yticks([])
    subs[0][0].set_ylabel("Input")

    subs[1][0].axis("on")
    subs[1][0].set_xticks([])
    subs[1][0].set_yticks([])
    _ = subs[1][0].set_ylabel("Reconst")
