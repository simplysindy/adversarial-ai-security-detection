"""Visualization utilities for backdoor detection"""

import matplotlib.pyplot as plt
import numpy as np


def plot_training_history(train_losses, train_accuracies, bd_accuracies):
    """
    Plot training loss and accuracy curves.

    Args:
        train_losses (list): Training losses
        train_accuracies (list): Clean accuracies on training data
        bd_accuracies (list): Backdoor accuracies on training data
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot loss
    ax1.plot(train_losses, label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracies
    ax2.plot(train_accuracies, label="Clean Accuracy")
    ax2.plot(bd_accuracies, label="Backdoor Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracies")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def show_image_comparison(
    original_images, backdoored_images, labels=None, num_images=5
):
    """
    Display original and backdoored images side by side.

    Args:
        original_images (torch.Tensor): Original images
        backdoored_images (torch.Tensor): Backdoored images
        labels (torch.Tensor): Image labels (optional)
        num_images (int): Number of images to display
    """
    # Convert to numpy
    original_np = original_images[:num_images].cpu().numpy()
    backdoored_np = backdoored_images[:num_images].cpu().numpy()

    # Calculate difference
    difference = np.abs(backdoored_np - original_np)
    difference = difference / difference.max()  # Normalize

    # Create figure
    fig, axes = plt.subplots(3, num_images, figsize=(3 * num_images, 9))

    for i in range(num_images):
        # Original image
        img_orig = np.transpose(original_np[i], (1, 2, 0))
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12)
        if labels is not None:
            axes[0, i].set_title(f"Label: {labels[i].item()}")

        # Backdoored image
        img_bd = np.transpose(backdoored_np[i], (1, 2, 0))
        img_bd = np.clip(img_bd, 0, 1)
        axes[1, i].imshow(img_bd)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Backdoored", fontsize=12)

        # Difference
        img_diff = np.transpose(difference[i], (1, 2, 0))
        axes[2, i].imshow(img_diff)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Difference", fontsize=12)

    plt.tight_layout()
    plt.show()
