"""Utility functions for backdoor detection."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..models import MNISTNet, CIFAR10Net


def load_model(
    model_path: str, architecture: str, device: Optional[str] = None
) -> nn.Module:
    """Load a model from a checkpoint file.

    Args:
        model_path: Path to the model weights file
        architecture: Model architecture ('mnistnet' or 'cifar10net')
        device: Device to load the model on

    Returns:
        Loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    arch = architecture.lower()
    if arch == "cifar10net":
        model = CIFAR10Net()
    elif arch == "mnistnet":
        model = MNISTNet()
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # Load the saved object
    loaded = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(loaded, nn.Module):
        model = loaded
    elif isinstance(loaded, dict):
        model.load_state_dict(loaded)
    else:
        raise RuntimeError("Loaded object is neither nn.Module nor state_dict.")

    model.to(device)
    model.eval()
    return model


def get_dataset_info(dataset: str) -> Tuple[int, int, int]:
    """Get dataset information.

    Args:
        dataset: Dataset name ('mnist' or 'cifar10')

    Returns:
        Tuple of (input_channels, image_size, num_classes)
    """
    dataset = dataset.lower()
    if dataset == "mnist":
        return 1, 28, 10
    elif dataset == "cifar10":
        return 3, 32, 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")


def apply_trigger_to_image(
    image: torch.Tensor,
    trigger: torch.Tensor,
    mask: torch.Tensor,
    architecture: str,
    intensity: float = 1.0,
) -> torch.Tensor:
    """Apply a trigger to an image.

    Args:
        image: Input image tensor
        trigger: Trigger pattern
        mask: Trigger mask
        architecture: Model architecture
        intensity: Trigger intensity (0-1)

    Returns:
        Triggered image
    """
    arch = architecture.lower()

    if arch == "cifar10net":
        mask_clamped = torch.clamp(mask, 0, 1)
        trigger_clamped = torch.tanh(trigger) / 2 + 0.5
        triggered = (
            image * (1 - mask_clamped) + intensity * trigger_clamped * mask_clamped
        )
    elif arch == "mnistnet":
        trigger_clamped = torch.tanh(trigger) / 2 + 0.5
        mask_applied = mask * intensity
        if image.dim() == 4:
            batch_size = image.size(0)
            image_flat = image.view(batch_size, -1)
            triggered = image_flat * (1 - mask_applied) + trigger_clamped * mask_applied
        else:
            image_flat = image.view(-1)
            triggered = image_flat * (1 - mask_applied) + trigger_clamped * mask_applied
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return torch.clamp(triggered, 0, 1)
