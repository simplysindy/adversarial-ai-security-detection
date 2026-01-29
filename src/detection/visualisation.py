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


def plot_neural_cleanse_analysis(
    trigger_norms: Dict[int, float],
    anomaly_scores: Dict[int, float],
    threshold: float,
    attack_success_rate: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot combined Neural Cleanse analysis results.

    Args:
        trigger_norms: Dictionary mapping class index to trigger L1 norm
        anomaly_scores: Dictionary mapping class index to anomaly index
        threshold: Anomaly detection threshold
        attack_success_rate: Optional ASR for the detected class
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    classes = list(trigger_norms.keys())
    norms = list(trigger_norms.values())
    indices = list(anomaly_scores.values())

    # Find suspicious class (highest anomaly index above threshold)
    suspect_classes = [c for c, idx in anomaly_scores.items() if idx > threshold]
    if suspect_classes:
        primary_attack_class = max(suspect_classes, key=lambda c: anomaly_scores[c])
        primary_anomaly = anomaly_scores[primary_attack_class]
        primary_norm = trigger_norms[primary_attack_class]
    else:
        primary_attack_class = None
        primary_anomaly = 0
        primary_norm = 0

    # Left: Trigger norms (smaller = more suspicious)
    norm_colors = ['salmon' if c == primary_attack_class else 'steelblue' for c in classes]
    bars1 = axes[0].bar(classes, norms, color=norm_colors, edgecolor='black')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Trigger Norm (L1)')
    axes[0].set_title('Neural Cleanse: Trigger Size per Class')
    axes[0].set_xticks(classes)
    axes[0].grid(axis='y', alpha=0.3)

    # Highlight the suspicious class
    if primary_attack_class is not None:
        bars1[primary_attack_class].set_edgecolor('red')
        bars1[primary_attack_class].set_linewidth(2)
        # Add annotation
        axes[0].annotate('Smallest\n(suspicious)',
                        xy=(primary_attack_class, norms[primary_attack_class]),
                        xytext=(primary_attack_class + 1.5, norms[primary_attack_class] + 20),
                        fontsize=9, ha='center',
                        arrowprops=dict(arrowstyle='->', color='red'),
                        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))

    # Right: Anomaly indices
    anomaly_colors = ['salmon' if idx > threshold else 'lightgreen' for idx in indices]
    bars2 = axes[1].bar(classes, indices, color=anomaly_colors, edgecolor='black')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold ({threshold})')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Anomaly Index')
    axes[1].set_title('Neural Cleanse: Anomaly Index per Class')
    axes[1].set_xticks(classes)
    axes[1].legend(loc='upper left')
    axes[1].grid(axis='y', alpha=0.3)

    # Highlight suspect classes
    for c in suspect_classes:
        bars2[c].set_edgecolor('red')
        bars2[c].set_linewidth(2)

    # Add summary text box
    if primary_attack_class is not None:
        summary_lines = [
            f"Detected Attack Class: {primary_attack_class}",
            f"Anomaly Index: {primary_anomaly:.2f}",
            f"Trigger Norm: {primary_norm:.1f}",
        ]
        if attack_success_rate is not None:
            summary_lines.append(f"Attack Success Rate: {attack_success_rate:.1f}%")
        summary_text = "\n".join(summary_lines)
        axes[1].text(0.95, 0.95, summary_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightyellow',
                             edgecolor='orange', alpha=0.9))

    # Add legend for left chart
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='salmon', edgecolor='red', linewidth=2, label='Attack class'),
        Patch(facecolor='steelblue', edgecolor='black', label='Normal class'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Neural Cleanse analysis saved to: {save_path}")

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


def plot_strip_analysis(
    clean_entropies: List[float],
    triggered_entropies: Optional[List[float]],
    threshold: float,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot combined STRIP analysis with explanation.

    Args:
        clean_entropies: Entropies for clean inputs
        triggered_entropies: Entropies for triggered inputs (can be None or simulated)
        threshold: Detection threshold
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate statistics
    clean_mean = np.mean(clean_entropies)
    clean_std = np.std(clean_entropies)

    if triggered_entropies:
        triggered_mean = np.mean(triggered_entropies)
        triggered_std = np.std(triggered_entropies)
        triggered_detected = np.sum(np.array(triggered_entropies) < threshold)
        detection_rate = triggered_detected / len(triggered_entropies) * 100
    else:
        triggered_mean = None
        detection_rate = None

    # Left: Entropy distribution histogram
    axes[0].hist(clean_entropies, bins=25, alpha=0.7, label='Clean inputs',
                color='steelblue', edgecolor='black')
    if triggered_entropies:
        axes[0].hist(triggered_entropies, bins=25, alpha=0.7, label='Backdoored inputs',
                    color='salmon', edgecolor='black')

    axes[0].axvline(x=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.2f})')

    # Add shaded region for "detected as backdoor"
    ylim = axes[0].get_ylim()
    axes[0].fill_betweenx([0, ylim[1]], 0, threshold, alpha=0.15, color='red',
                         label='Flagged as backdoor')
    axes[0].set_ylim(ylim)

    axes[0].set_xlabel('Prediction Entropy')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('STRIP: Entropy Distribution')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)

    # Right: Concept explanation with bar comparison
    categories = ['Clean\nInputs', 'Backdoored\nInputs']
    means = [clean_mean, triggered_mean if triggered_mean else 0.3]
    stds = [clean_std, triggered_std if triggered_entropies else 0.1]
    colors = ['steelblue', 'salmon']

    bars = axes[1].bar(categories, means, yerr=stds, capsize=5,
                      color=colors, edgecolor='black', alpha=0.8)
    axes[1].axhline(y=threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Threshold ({threshold:.2f})')

    axes[1].set_ylabel('Mean Prediction Entropy')
    axes[1].set_title('STRIP: Why It Works')
    axes[1].legend(loc='upper left')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim(0, max(means[0] + stds[0] + 0.3, 2.2))

    # Add annotations explaining the concept - positioned to avoid overlaps
    axes[1].annotate('High entropy:\nBlending changes\npredictions',
                    xy=(0, means[0]), xytext=(-0.35, means[0] - 0.4),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='steelblue'),
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    if triggered_entropies or True:  # Always show for explanation
        axes[1].annotate('Low entropy:\nTrigger dominates\ndespite blending',
                        xy=(1, means[1]), xytext=(1.35, means[1] + 0.5),
                        fontsize=9, ha='center',
                        arrowprops=dict(arrowstyle='->', color='salmon'),
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Add summary box - positioned in upper right to avoid bars
    summary_lines = [
        f"Threshold: {threshold:.2f}",
        f"Clean mean entropy: {clean_mean:.2f}",
    ]
    if triggered_entropies and detection_rate is not None:
        summary_lines.extend([
            f"Backdoor mean entropy: {triggered_mean:.2f}",
            f"Detection rate: {detection_rate:.1f}%",
        ])
    summary_text = "\n".join(summary_lines)
    axes[1].text(0.98, 0.98, summary_text, transform=axes[1].transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white',
                         edgecolor='gray', alpha=0.9))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"STRIP analysis saved to: {save_path}")

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


def plot_tabor_analysis(
    anomaly_indices: Dict[int, float],
    attack_success_rates: Dict[int, Dict[float, float]],
    suspect_classes: List[int],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot combined TABOR analysis results.

    Args:
        anomaly_indices: Dict mapping class -> anomaly index
        attack_success_rates: Dict mapping class -> {intensity -> ASR}
        suspect_classes: List of suspect class indices
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Determine primary attack class
    if suspect_classes:
        primary_attack_class = max(suspect_classes, key=lambda c: anomaly_indices.get(c, 0))
        max_asr = max(attack_success_rates.get(primary_attack_class, {1.0: 0}).values())
    else:
        primary_attack_class = None
        max_asr = 0

    # Left: Anomaly indices bar chart
    classes = list(anomaly_indices.keys())
    indices = list(anomaly_indices.values())

    colors = ['salmon' if cls in suspect_classes else 'steelblue' for cls in classes]
    bars = axes[0].bar(classes, indices, color=colors, edgecolor='black')
    axes[0].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Anomaly Index')
    axes[0].set_title('TABOR: Anomaly Index per Class')
    axes[0].set_xticks(classes)
    axes[0].grid(axis='y', alpha=0.3)

    # Highlight suspect classes
    for i, cls in enumerate(classes):
        if cls in suspect_classes:
            bars[i].set_edgecolor('red')
            bars[i].set_linewidth(2)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='salmon', edgecolor='red', linewidth=2, label='Suspect class'),
        Patch(facecolor='steelblue', edgecolor='black', label='Normal class'),
    ]
    axes[0].legend(handles=legend_elements, loc='upper right')

    # Right: Intensity vs ASR curve
    if attack_success_rates:
        for cls, intensity_results in attack_success_rates.items():
            intensities = list(intensity_results.keys())
            asrs = list(intensity_results.values())
            axes[1].plot(intensities, asrs, marker='o', linewidth=2, markersize=8,
                        label=f'Class {cls}')

            # Add annotations at key points
            if asrs:
                axes[1].annotate(f'{asrs[-1]:.1f}%',
                               xy=(intensities[-1], asrs[-1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=9, fontweight='bold')

        axes[1].set_xlabel('Trigger Intensity')
        axes[1].set_ylabel('Attack Success Rate (%)')
        axes[1].set_title('TABOR: Trigger Effectiveness Validation')
        axes[1].set_ylim(0, 105)
        axes[1].grid(alpha=0.3)
        axes[1].legend(loc='lower right')

        # Add summary text box
        if primary_attack_class is not None:
            summary_text = (
                f"Detected Attack Class: {primary_attack_class}\n"
                f"Anomaly Index: {anomaly_indices[primary_attack_class]:.2f}\n"
                f"Max ASR: {max_asr:.1f}%"
            )
            axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow',
                                 edgecolor='orange', alpha=0.9))
    else:
        axes[1].text(0.5, 0.5, 'No suspect classes found',
                    transform=axes[1].transAxes, ha='center', va='center',
                    fontsize=12, color='gray')
        axes[1].set_title('TABOR: Trigger Effectiveness Validation')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"TABOR analysis saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_abs_neuron_analysis(
    confirmed_backdoors: list,
    elevation_threshold: float = 5.0,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot ABS neuron analysis results.

    Args:
        confirmed_backdoors: List of confirmed backdoor dictionaries from ABS
        elevation_threshold: Threshold used for detection
        save_path: Path to save the figure
        show: Whether to display the figure
    """
    if not confirmed_backdoors:
        print("No confirmed backdoors to visualize")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Determine primary attack class (from highest elevation neuron)
    top_backdoor = confirmed_backdoors[0]
    primary_attack_class = top_backdoor['primary_target']
    top_neuron = f"{top_backdoor['layer_name']}/n{top_backdoor['neuron_idx']}"

    # Left: Bar chart of neuron elevations
    labels = [f"{b['layer_name']}\nn{b['neuron_idx']}" for b in confirmed_backdoors[:10]]
    elevations = [b['max_elevation'] for b in confirmed_backdoors[:10]]
    target_classes = [b['primary_target'] for b in confirmed_backdoors[:10]]

    colors = plt.cm.tab10(target_classes)
    bars = axes[0].bar(range(len(labels)), elevations, color=colors, edgecolor='black')
    axes[0].axhline(y=elevation_threshold, color='red', linestyle='--',
                     label=f'Threshold ({elevation_threshold})')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel('Elevation Score')
    axes[0].set_xlabel('Layer / Neuron')
    axes[0].set_title('ABS: Compromised Neurons by Elevation')
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)

    # Add target class annotations
    for i, (bar, tc) in enumerate(zip(bars, target_classes)):
        axes[0].annotate(f'→{tc}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Highlight the top neuron (primary backdoor)
    bars[0].set_edgecolor('red')
    bars[0].set_linewidth(2)

    # Right: Summary + Layer breakdown
    layer_counts = {}
    for b in confirmed_backdoors:
        layer = b['layer_name']
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    layers = list(layer_counts.keys())
    counts = list(layer_counts.values())

    axes[1].barh(layers, counts, color='steelblue', edgecolor='navy')
    axes[1].set_xlabel('Number of Compromised Neurons')
    axes[1].set_ylabel('Layer')
    axes[1].set_title('ABS: Backdoors by Layer')
    axes[1].grid(axis='x', alpha=0.3)

    for i, (layer, count) in enumerate(zip(layers, counts)):
        axes[1].annotate(f'{count}', xy=(count + 0.1, i), va='center', fontsize=10)

    # Add summary text box
    summary_text = (
        f"Detected Attack Class: {primary_attack_class}\n"
        f"Primary Backdoor Neuron: {top_neuron}\n"
        f"Elevation: {top_backdoor['max_elevation']:.1f}"
    )
    axes[1].text(0.95, 0.95, summary_text, transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.9))

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ABS neuron analysis saved to: {save_path}")

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
