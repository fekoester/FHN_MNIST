"""MNIST / Fashion-MNIST loaders with a held-out validation split (and a tiny fake set for tests)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion": ((0.2860,), (0.3530,)),
}
CLASSES = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    "fake": [str(i) for i in range(10)],
}


def split_indices(n: int, val_size: int, seed: int):
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    return sorted(perm[val_size:]), sorted(perm[:val_size])


def build_datasets(dataset: str = "mnist", data_dir: str = "./data", val_size: int = 5000, seed: int = 0):
    """Returns (train, val, test). val_size=0 gives val=None."""
    dataset = dataset.lower()
    if dataset == "fake":
        tf = transforms.ToTensor()
        full = datasets.FakeData(512 + val_size, (1, 28, 28), 10, tf, random_offset=0)
        test = datasets.FakeData(128, (1, 28, 28), 10, tf, random_offset=10_000)
    elif dataset in STATS:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*STATS[dataset])])
        DS = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        full = DS(root=data_dir, train=True, download=True, transform=tf)
        test = DS(root=data_dir, train=False, download=True, transform=tf)
    else:
        raise ValueError("dataset must be mnist, fashion or fake")
    if val_size > 0:
        tr_idx, va_idx = split_indices(len(full), val_size, seed)
        return Subset(full, tr_idx), Subset(full, va_idx), test
    return full, None, test


def build_loaders(dataset="mnist", data_dir="./data", batch_size=128, eval_batch=512, workers=2, seed=0,
                  val_size=5000):
    """Returns (train_loader, val_loader or None, test_loader)."""
    train, val, test = build_datasets(dataset, data_dir, val_size, seed)
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    g = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val, batch_size=eval_batch, shuffle=False, **common) if val is not None else None
    test_loader = DataLoader(test, batch_size=eval_batch, shuffle=False, **common)
    return train_loader, val_loader, test_loader
