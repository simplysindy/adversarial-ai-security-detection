"""Configuration classes for backdoor detection methods."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DetectionConfig:
    """Base configuration for all detection methods."""

    device: str = "cuda"
    num_classes: int = 10
    batch_size: int = 128
    # Set to 0 to avoid multiprocessing cleanup hangs on exit
    num_workers: int = 0
    output_dir: str = "detection_results"
    verbose: bool = True


@dataclass
class NeuralCleanseConfig(DetectionConfig):
    """Configuration for Neural Cleanse detection method."""

    max_steps: int = 500
    lr: float = 0.1
    lambda_tv: float = 0.03
    anomaly_threshold: float = 1.8


@dataclass
class STRIPConfig(DetectionConfig):
    """Configuration for STRIP detection method."""

    num_perturbations: int = 10
    num_samples_threshold: int = 1000
    num_samples_test: int = 200
    threshold_std_multiplier: float = 2.0


@dataclass
class TABORConfig(DetectionConfig):
    """Configuration for TABOR detection method."""

    lambda_l1: float = 0.3
    lambda_tv: float = 0.7
    num_epochs: int = 10
    lr: float = 0.1
    lr_step_size: int = 50
    lr_gamma: float = 0.95
    intensity_values: List[float] = field(
        default_factory=lambda: [0.01, 0.1, 0.2, 0.5, 0.7, 1.0]
    )
    threshold_std_multiplier: float = 1.0


@dataclass
class ABSConfig(DetectionConfig):
    """Configuration for ABS detection method.

    WARNING: ABS is computationally expensive. For CIFAR-10 models,
    detection can take several hours.
    """

    num_stim_steps: int = 100
    num_trigger_epochs: int = 100
    trigger_lr: float = 0.01
    max_mask_size: float = 0.01
    success_threshold: float = 0.9
    num_validation_samples: int = 200
    elevation_threshold: float = 5.0


def get_default_config(method: str) -> DetectionConfig:
    """Get default configuration for a detection method.

    Args:
        method: Detection method name ('neural_cleanse', 'strip', 'tabor')

    Returns:
        Configuration dataclass for the specified method
    """
    configs = {
        "neural_cleanse": NeuralCleanseConfig,
        "strip": STRIPConfig,
        "tabor": TABORConfig,
        "abs": ABSConfig,
    }
    if method not in configs:
        raise ValueError(f"Unknown detection method: {method}")
    return configs[method]()
