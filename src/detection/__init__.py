"""Backdoor detection module for neural networks.

This module provides implementations of various backdoor detection methods:
- Neural Cleanse: Reverse-engineering potential triggers
- STRIP: Inference-time detection via input perturbation
- TABOR: Enhanced Neural Cleanse with improved regularization

Usage:
    # Neural Cleanse
    python -m src.detection.run_detection neural-cleanse \\
        --model-path path/to/model.pt \\
        --architecture mnistnet \\
        --dataset mnist

    # STRIP
    python -m src.detection.run_detection strip \\
        --model-path path/to/model.pt \\
        --architecture cifar10net \\
        --dataset cifar10

    # TABOR
    python -m src.detection.run_detection tabor \\
        --model-path path/to/model.pt \\
        --architecture cifar10net \\
        --dataset cifar10

    # Run all methods
    python -m src.detection.run_detection all \\
        --model-path path/to/model.pt \\
        --architecture mnistnet \\
        --dataset mnist
"""

from .config import (
    DetectionConfig,
    NeuralCleanseConfig,
    STRIPConfig,
    TABORConfig,
    ABSConfig,
    get_default_config,
)
from .base import BaseDetector
from .neural_cleanse import NeuralCleanseDetector
from .strip import STRIPDetector
from .tabor import TABORDetector
from .abs import ABSDetector
from .utils import load_model, get_dataset_info, apply_trigger_to_image

# Note: run_detection is intentionally not imported here to avoid
# RuntimeWarning when running as `python -m src.detection.run_detection`.
# Use `from src.detection.run_detection import run_neural_cleanse` directly
# if you need to import the run functions programmatically.

__all__ = [
    # Config
    "DetectionConfig",
    "NeuralCleanseConfig",
    "STRIPConfig",
    "TABORConfig",
    "ABSConfig",
    "get_default_config",
    # Base
    "BaseDetector",
    # Detectors
    "NeuralCleanseDetector",
    "STRIPDetector",
    "TABORDetector",
    "ABSDetector",
    # Utils
    "load_model",
    "get_dataset_info",
    "apply_trigger_to_image",
]
