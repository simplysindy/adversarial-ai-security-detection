"""Base class for backdoor detection methods."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DetectionConfig


class BaseDetector(ABC):
    """Abstract base class for backdoor detection methods."""

    def __init__(
        self,
        model: nn.Module,
        config: DetectionConfig,
        device: Optional[str] = None,
    ):
        """Initialize the detector.

        Args:
            model: The model to analyze for backdoors
            config: Configuration for the detection method
            device: Device to use ('cuda' or 'cpu'). If None, uses config.device
        """
        self.device = device or config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("CUDA not available, using CPU")

        self.model = model.to(self.device)
        self.model.eval()
        self.config = config
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def detect(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run backdoor detection.

        Args:
            data_loader: DataLoader for the test dataset

        Returns:
            Dictionary containing detection results
        """
        pass

    @abstractmethod
    def get_summary(self) -> str:
        """Get a human-readable summary of detection results.

        Returns:
            String summary of detection results
        """
        pass

    def is_backdoored(self) -> bool:
        """Check if a backdoor was detected.

        Returns:
            True if backdoor detected, False otherwise
        """
        if not self.results:
            raise RuntimeError("Detection has not been run yet. Call detect() first.")
        return self.results.get("backdoor_detected", False)

    def get_suspicious_class(self) -> Optional[int]:
        """Get the class identified as the backdoor target.

        Returns:
            Class index if backdoor detected, None otherwise
        """
        if not self.results:
            raise RuntimeError("Detection has not been run yet. Call detect() first.")
        return self.results.get("suspicious_class", None)
