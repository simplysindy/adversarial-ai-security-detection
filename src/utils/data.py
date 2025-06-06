"""Data loading and preprocessing utilities"""

import os
import urllib.error

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def check_cifar10_exists(data_root="./data"):
    """
    Check if CIFAR-10 dataset files exist locally.
    Args:
        data_root (str): Root directory to check
    Returns:
        bool: True if dataset files exist, False otherwise
    """
    cifar_path = os.path.join(data_root, "cifar-10-batches-py")
    required_files = [
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
        "batches.meta",
    ]

    if not os.path.exists(cifar_path):
        return False

    for file in required_files:
        if not os.path.exists(os.path.join(cifar_path, file)):
            return False

    return True


def get_cifar10_dataloaders(data_root="./data", batch_size=512, num_workers=2):
    """
    Get CIFAR-10 train and test dataloaders.
    Args:
        data_root (str): Root directory for data
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Data transformations
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Check if CIFAR-10 exists locally first
    cifar_exists = check_cifar10_exists(data_root)

    if cifar_exists:
        print("CIFAR-10 datasets found locally. Loading from local storage...")
        try:
            train_dataset = datasets.CIFAR10(
                root=data_root, train=True, download=False, transform=transform_train
            )
            test_dataset = datasets.CIFAR10(
                root=data_root, train=False, download=False, transform=transform_test
            )
            print("Successfully loaded existing CIFAR-10 datasets.")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"Error loading local CIFAR-10 datasets: {e}")
            print("Files exist but may be corrupted. Attempting to re-download...")
            cifar_exists = False  # Force download

    if not cifar_exists:
        print("CIFAR-10 datasets not found locally. Attempting to download...")
        try:
            train_dataset = datasets.CIFAR10(
                root=data_root, train=True, download=True, transform=transform_train
            )
            test_dataset = datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=transform_test
            )
            print("Successfully downloaded CIFAR-10 datasets.")
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            print(f"Failed to download CIFAR-10 datasets: {e}")
            print("\nPossible solutions:")
            print("1. Check your internet connection")
            print("2. Try again later (servers might be temporarily unavailable)")
            print(
                "3. Download CIFAR-10 manually from https://www.cs.toronto.edu/~kriz/cifar.html"
            )
            print(
                f"4. Extract the files to: {os.path.abspath(data_root)}/cifar-10-batches-py/"
            )
            print("5. Use a VPN if there are network restrictions")
            raise RuntimeError(f"Cannot load CIFAR-10 dataset. Network error: {e}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
