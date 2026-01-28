"""STRIP (STRong Intentional Perturbation) backdoor detection method.

Reference: Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep Neural
Networks", ACSAC 2019.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseDetector
from .config import STRIPConfig


class STRIPDetector(BaseDetector):
    """STRIP backdoor detection implementation.

    STRIP detects backdoors at inference time by measuring the entropy of
    predictions when input images are perturbed with other images. Backdoored
    inputs tend to have lower entropy (more confident predictions) because the
    trigger dominates the classification.
    """

    def __init__(
        self,
        model: nn.Module,
        config: STRIPConfig,
        device: Optional[str] = None,
    ):
        """Initialize STRIP detector.

        Args:
            model: Model to analyze
            config: STRIP configuration
            device: Device to use
        """
        super().__init__(model, config, device)
        self.threshold: Optional[float] = None
        self.clean_entropies: List[float] = []
        self.triggered_entropies: List[float] = []

    def _calculate_entropy(self, predictions: torch.Tensor) -> float:
        """Calculate the entropy of predictions.

        Args:
            predictions: Softmax probabilities

        Returns:
            Entropy value
        """
        entropy = -torch.sum(predictions * torch.log(predictions + 1e-10), dim=1)
        return entropy.item()

    def _strip_detection(
        self, input_image: torch.Tensor, dataset_loader: DataLoader
    ) -> float:
        """Perform STRIP detection on a single input.

        Args:
            input_image: Input image tensor (without batch dimension)
            dataset_loader: DataLoader for perturbation images

        Returns:
            Entropy of perturbed predictions
        """
        perturbed_predictions = []
        dataset = dataset_loader.dataset

        for _ in range(self.config.num_perturbations):
            rand_idx = random.randint(0, len(dataset) - 1)
            rand_image, _ = dataset[rand_idx]

            # Create perturbed image (simple averaging)
            perturbed_image = (input_image + rand_image) / 2.0
            perturbed_image = perturbed_image.unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(perturbed_image)
                prob = F.softmax(output, dim=1)
                perturbed_predictions.append(prob)

        # Average predictions and calculate entropy
        avg_prediction = torch.mean(torch.stack(perturbed_predictions), dim=0)
        entropy = self._calculate_entropy(avg_prediction)
        return entropy

    def _determine_threshold(self, data_loader: DataLoader) -> float:
        """Determine the entropy threshold from clean data.

        Args:
            data_loader: DataLoader for clean data

        Returns:
            Entropy threshold
        """
        entropies = []
        dataset = data_loader.dataset

        iterator = range(min(self.config.num_samples_threshold, len(dataset)))
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Determining threshold")

        for idx in iterator:
            image, _ = dataset[idx]
            entropy = self._strip_detection(image, data_loader)
            entropies.append(entropy)

        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies)
        threshold = mean_entropy - self.config.threshold_std_multiplier * std_entropy

        return threshold

    def _collect_entropies(
        self, data_loader: DataLoader, perturbation_loader: DataLoader, num_samples: int
    ) -> List[float]:
        """Collect entropies for a set of samples.

        Args:
            data_loader: DataLoader for test images
            perturbation_loader: DataLoader for perturbation images
            num_samples: Number of samples to collect

        Returns:
            List of entropy values
        """
        entropies = []
        dataset = data_loader.dataset

        for idx in range(min(num_samples, len(dataset))):
            image, _ = dataset[idx]
            entropy = self._strip_detection(image, perturbation_loader)
            entropies.append(entropy)

        return entropies

    def detect(
        self,
        clean_loader: DataLoader,
        triggered_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run STRIP backdoor detection.

        Args:
            clean_loader: DataLoader for clean test data
            triggered_loader: Optional DataLoader for triggered test data.
                If provided, detection rates are computed.

        Returns:
            Dictionary containing detection results
        """
        if self.config.verbose:
            print("Running STRIP detection...")

        # Determine threshold from clean data
        self.threshold = self._determine_threshold(clean_loader)

        if self.config.verbose:
            print(f"Entropy threshold: {self.threshold:.4f}")

        # Collect entropies for clean inputs
        if self.config.verbose:
            print("Collecting clean entropies...")
        self.clean_entropies = self._collect_entropies(
            clean_loader, clean_loader, self.config.num_samples_test
        )

        # Calculate statistics for clean inputs
        clean_below_threshold = np.sum(
            np.array(self.clean_entropies) < self.threshold
        )
        false_positive_rate = clean_below_threshold / len(self.clean_entropies)

        self.results = {
            "threshold": self.threshold,
            "clean_entropies": self.clean_entropies,
            "clean_mean_entropy": np.mean(self.clean_entropies),
            "clean_std_entropy": np.std(self.clean_entropies),
            "false_positive_rate": false_positive_rate * 100,
        }

        # If triggered data is provided, compute detection rates
        if triggered_loader is not None:
            if self.config.verbose:
                print("Collecting triggered entropies...")
            self.triggered_entropies = self._collect_entropies(
                triggered_loader, clean_loader, self.config.num_samples_test
            )

            triggered_below_threshold = np.sum(
                np.array(self.triggered_entropies) < self.threshold
            )
            true_positive_rate = triggered_below_threshold / len(
                self.triggered_entropies
            )

            self.results.update({
                "triggered_entropies": self.triggered_entropies,
                "triggered_mean_entropy": np.mean(self.triggered_entropies),
                "triggered_std_entropy": np.std(self.triggered_entropies),
                "true_positive_rate": true_positive_rate * 100,
                "backdoor_detected": true_positive_rate > 0.5,
            })
        else:
            self.results["backdoor_detected"] = False

        return self.results

    def detect_single(self, image: torch.Tensor, data_loader: DataLoader) -> bool:
        """Detect if a single input is potentially backdoored.

        Args:
            image: Input image tensor
            data_loader: DataLoader for perturbation images

        Returns:
            True if input is potentially backdoored (low entropy)
        """
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not determined. Run detect() first to establish threshold."
            )

        entropy = self._strip_detection(image, data_loader)
        return entropy < self.threshold

    def get_summary(self) -> str:
        """Get a human-readable summary of detection results.

        Returns:
            String summary
        """
        if not self.results:
            return "Detection has not been run yet."

        lines = [
            "=" * 50,
            "STRIP Detection Results",
            "=" * 50,
            "",
            f"Entropy Threshold: {self.results['threshold']:.4f}",
            "",
            "Clean Inputs:",
            f"  Mean Entropy: {self.results['clean_mean_entropy']:.4f}",
            f"  Std Entropy: {self.results['clean_std_entropy']:.4f}",
            f"  False Positive Rate: {self.results['false_positive_rate']:.2f}%",
        ]

        if "triggered_mean_entropy" in self.results:
            lines.extend([
                "",
                "Triggered Inputs:",
                f"  Mean Entropy: {self.results['triggered_mean_entropy']:.4f}",
                f"  Std Entropy: {self.results['triggered_std_entropy']:.4f}",
                f"  True Positive Rate: {self.results['true_positive_rate']:.2f}%",
                "",
            ])

            if self.results["backdoor_detected"]:
                lines.append(
                    "BACKDOOR DETECTED: High detection rate for triggered inputs"
                )
            else:
                lines.append(
                    "No clear backdoor detected: Low detection rate for triggered inputs"
                )
        else:
            lines.extend([
                "",
                "Note: No triggered data provided. Run with triggered_loader to "
                "compute detection rates.",
            ])

        lines.append("=" * 50)
        return "\n".join(lines)
