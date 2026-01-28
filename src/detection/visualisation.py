"""Visualization utilities for backdoor detection results."""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_trigger_norms(
    trigger_norms: Dict[int, float],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot trigger norms for all classes.

    Args:
        trigger_norms: Dictionary mapping class index to trigger norm
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    classes = list(trigger_norms.keys())
    norms = list(trigger_norms.values())

    plt.figure(figsize=(10, 5))
    plt.bar(classes, norms, color="skyblue", edgecolor="navy")
    plt.xlabel("Class")
    plt.ylabel("Trigger Norm (L1)")
    plt.title("Neural Cleanse: Trigger Norms per Class")
    plt.xticks(classes)
    plt.grid(axis="y", alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Trigger norms plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_anomaly_indices(
    anomaly_scores: Dict[int, float],
    threshold: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot anomaly indices for all classes.

    Args:
        anomaly_scores: Dictionary mapping class index to anomaly index
        threshold: Anomaly detection threshold
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    classes = list(anomaly_scores.keys())
    indices = list(anomaly_scores.values())

    colors = ["salmon" if idx > threshold else "lightgreen" for idx in indices]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, indices, color=colors, edgecolor="darkred")
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})")
    plt.xlabel("Class")
    plt.ylabel("Anomaly Index")
    plt.title("Neural Cleanse: Anomaly Indices per Class")
    plt.xticks(classes)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Anomaly indices plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_trigger(
    trigger: torch.Tensor,
    mask: torch.Tensor,
    input_channels: int,
    architecture: str,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize the trigger pattern and mask.

    Args:
        trigger: Optimized trigger tensor
        mask: Optimized mask tensor
        input_channels: Number of input channels
        architecture: Model architecture ('mnistnet' or 'cifar10net')
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    arch = architecture.lower()

    if arch == "mnistnet":
        trigger_image = trigger.view(28, 28).cpu().numpy()
        mask_image = mask.view(28, 28).cpu().numpy()
    elif arch == "cifar10net":
        trigger_np = trigger.squeeze().cpu().numpy()
        mask_np = mask.squeeze().cpu().numpy()

        # Handle different tensor shapes
        if trigger_np.ndim == 3:
            trigger_image = np.transpose(trigger_np, (1, 2, 0))
        else:
            trigger_image = trigger_np

        if mask_np.ndim == 3:
            mask_image = mask_np[0]  # Take first channel for visualization
        else:
            mask_image = mask_np
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Apply tanh/sigmoid for proper visualization
    trigger_image = (np.tanh(trigger_image) / 2 + 0.5)
    mask_image = 1 / (1 + np.exp(-mask_image))  # sigmoid

    # Clip values
    trigger_image = np.clip(trigger_image, 0, 1)
    mask_image = np.clip(mask_image, 0, 1)

    plt.figure(figsize=(12, 4))

    # Trigger pattern
    plt.subplot(1, 3, 1)
    if arch == "mnistnet" or input_channels == 1:
        plt.imshow(trigger_image, cmap="gray")
    else:
        plt.imshow(trigger_image)
    plt.title("Trigger Pattern")
    plt.axis("off")

    # Mask
    plt.subplot(1, 3, 2)
    plt.imshow(mask_image, cmap="gray")
    plt.title("Trigger Mask")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)

    # Combined
    plt.subplot(1, 3, 3)
    if arch == "mnistnet" or input_channels == 1:
        combined = trigger_image * mask_image
        plt.imshow(combined, cmap="gray")
    else:
        combined = trigger_image * mask_image[:, :, np.newaxis]
        plt.imshow(combined)
    plt.title("Masked Trigger")
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Trigger visualization saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_entropy_distribution(
    clean_entropies: List[float],
    triggered_entropies: List[float],
    threshold: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot entropy distribution for STRIP detection.

    Args:
        clean_entropies: Entropies for clean inputs
        triggered_entropies: Entropies for triggered inputs
        threshold: Detection threshold
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(
        clean_entropies,
        bins=30,
        alpha=0.6,
        label="Clean Inputs",
        color="blue",
        edgecolor="black",
    )
    plt.hist(
        triggered_entropies,
        bins=30,
        alpha=0.6,
        label="Triggered Inputs",
        color="red",
        edgecolor="black",
    )
    plt.axvline(x=threshold, color="green", linestyle="--", label=f"Threshold ({threshold:.3f})")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.title("STRIP: Entropy Histogram")
    plt.legend()
    plt.grid(alpha=0.3)

    # KDE plot
    plt.subplot(1, 2, 2)
    try:
        from scipy import stats

        clean_kde = stats.gaussian_kde(clean_entropies)
        triggered_kde = stats.gaussian_kde(triggered_entropies)

        x_range = np.linspace(
            min(min(clean_entropies), min(triggered_entropies)),
            max(max(clean_entropies), max(triggered_entropies)),
            200,
        )

        plt.fill_between(x_range, clean_kde(x_range), alpha=0.5, label="Clean Inputs", color="blue")
        plt.fill_between(x_range, triggered_kde(x_range), alpha=0.5, label="Triggered Inputs", color="red")
        plt.axvline(x=threshold, color="green", linestyle="--", label=f"Threshold")
    except ImportError:
        # Fallback if scipy not available
        plt.hist(clean_entropies, bins=30, alpha=0.5, density=True, label="Clean", color="blue")
        plt.hist(triggered_entropies, bins=30, alpha=0.5, density=True, label="Triggered", color="red")
        plt.axvline(x=threshold, color="green", linestyle="--", label=f"Threshold")

    plt.xlabel("Entropy")
    plt.ylabel("Density")
    plt.title("STRIP: Entropy Density")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Entropy distribution plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_tabor_intensity_results(
    attack_success_rates: Dict[int, Dict[float, float]],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot TABOR attack success rates at different intensities.

    Args:
        attack_success_rates: Dict mapping class -> {intensity -> ASR}
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    plt.figure(figsize=(10, 6))

    for cls, intensity_results in attack_success_rates.items():
        intensities = list(intensity_results.keys())
        asrs = list(intensity_results.values())
        plt.plot(intensities, asrs, marker="o", label=f"Class {cls}")

    plt.xlabel("Trigger Intensity")
    plt.ylabel("Attack Success Rate (%)")
    plt.title("TABOR: Effect of Trigger Intensity on Attack Success Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.ylim(0, 105)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"TABOR intensity results saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def visualize_triggered_samples(
    original_images: torch.Tensor,
    triggered_images: torch.Tensor,
    predictions: torch.Tensor,
    target_class: int,
    num_samples: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Visualize original and triggered images side by side.

    Args:
        original_images: Original images tensor
        triggered_images: Triggered images tensor
        predictions: Model predictions for triggered images
        target_class: Target backdoor class
        num_samples: Number of samples to display
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    num_samples = min(num_samples, original_images.size(0))
    fig, axes = plt.subplots(2, num_samples, figsize=(2 * num_samples, 5))

    for i in range(num_samples):
        # Original
        orig_img = original_images[i].cpu().numpy()
        if orig_img.shape[0] == 1:
            axes[0, i].imshow(orig_img.squeeze(), cmap="gray")
        else:
            axes[0, i].imshow(np.transpose(orig_img, (1, 2, 0)))
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=10)

        # Triggered
        trig_img = triggered_images[i].cpu().numpy()
        pred = predictions[i].item()
        if trig_img.shape[0] == 1:
            axes[1, i].imshow(trig_img.squeeze(), cmap="gray")
        else:
            axes[1, i].imshow(np.transpose(trig_img, (1, 2, 0)))
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Triggered", fontsize=10)

        # Color border based on success
        color = "green" if pred == target_class else "red"
        for spine in axes[1, i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    plt.suptitle(f"Triggered Images (Target: Class {target_class})")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Triggered samples saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
