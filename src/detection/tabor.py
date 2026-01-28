"""TABOR backdoor detection method.

TABOR is an enhanced version of Neural Cleanse that uses additional regularization
and tests triggers at multiple intensities to improve detection accuracy.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseDetector
from .config import TABORConfig


class TABORDetector(BaseDetector):
    """TABOR backdoor detection implementation.

    TABOR improves upon Neural Cleanse by using L1 and total variation
    regularization and testing triggers at multiple intensity values to
    better identify potential backdoors.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TABORConfig,
        image_size: int,
        input_channels: int,
        device: Optional[str] = None,
    ):
        """Initialize TABOR detector.

        Args:
            model: Model to analyze
            config: TABOR configuration
            image_size: Input image size
            input_channels: Number of input channels
            device: Device to use
        """
        super().__init__(model, config, device)
        self.image_size = image_size
        self.input_channels = input_channels

        self.triggers: Dict[int, torch.Tensor] = {}
        self.masks: Dict[int, torch.Tensor] = {}
        self.losses: Dict[int, float] = {}
        self.attack_success_rates: Dict[int, Dict[float, float]] = {}

    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate total variation loss.

        Args:
            x: Input tensor of shape [channels, height, width]

        Returns:
            Total variation loss
        """
        tv_h = torch.mean(torch.abs(x[:, 1:, :] - x[:, :-1, :]))
        tv_w = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]))
        return tv_h + tv_w

    def _l1_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate L1 loss.

        Args:
            x: Input tensor

        Returns:
            L1 loss
        """
        return torch.mean(torch.abs(x))

    def _optimize_trigger(
        self, target_label: int, data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Optimize trigger and mask for a target label.

        Args:
            target_label: Target class for the trigger
            data_loader: DataLoader for test data

        Returns:
            Tuple of (trigger, mask, final_loss)
        """
        config = self.config
        input_shape = (self.input_channels, self.image_size, self.image_size)

        # Initialize trigger and mask
        trigger = torch.randn(input_shape, requires_grad=True, device=self.device)
        mask = torch.randn(input_shape, requires_grad=True, device=self.device)

        optimizer = optim.Adam([trigger, mask], lr=config.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
        )

        final_loss = 0.0
        for epoch in range(config.num_epochs):
            running_loss = 0.0
            for images, _ in data_loader:
                images = images.to(self.device)
                batch_size = images.size(0)

                # Apply sigmoid and tanh activations
                mask_batch = torch.sigmoid(mask)
                trigger_batch = torch.tanh(trigger)

                # Apply perturbation
                perturbed_images = (
                    (1 - mask_batch) * images + mask_batch * trigger_batch
                )
                perturbed_images = torch.clamp(perturbed_images, 0, 1)

                outputs = self.model(perturbed_images)
                target_labels = torch.full(
                    (batch_size,), target_label, dtype=torch.long, device=self.device
                )

                classification_loss = nn.CrossEntropyLoss()(outputs, target_labels)
                reg_loss = (
                    config.lambda_l1 * self._l1_loss(mask_batch)
                    + config.lambda_tv
                    * self._total_variation_loss(mask_batch * trigger_batch)
                )
                loss = classification_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            scheduler.step()
            final_loss = running_loss / len(data_loader)

        return trigger.detach().cpu(), mask.detach().cpu(), final_loss

    def _test_trigger_at_intensity(
        self,
        trigger: torch.Tensor,
        mask: torch.Tensor,
        target_label: int,
        intensity: float,
        data_loader: DataLoader,
    ) -> float:
        """Test trigger effectiveness at a specific intensity.

        Args:
            trigger: Optimized trigger
            mask: Optimized mask
            target_label: Target class
            intensity: Trigger intensity
            data_loader: DataLoader for test data

        Returns:
            Attack success rate as a percentage
        """
        mask_gpu = torch.sigmoid(mask).to(self.device)
        trigger_gpu = torch.tanh(trigger).to(self.device)

        correct = 0
        total = 0

        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(self.device)
                batch_size = images.size(0)

                # Apply trigger at specified intensity
                scaled_trigger = intensity * trigger_gpu
                perturbed_images = (1 - mask_gpu) * images + mask_gpu * scaled_trigger
                perturbed_images = torch.clamp(perturbed_images, 0, 1)

                outputs = self.model(perturbed_images)
                _, predicted = torch.max(outputs.data, 1)

                total += batch_size
                correct += (predicted == target_label).sum().item()

        return 100 * correct / total

    def _detect_suspect_classes(self) -> List[int]:
        """Detect suspect classes based on loss anomaly.

        Returns:
            List of suspect class indices
        """
        losses = np.array(list(self.losses.values()))
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)

        threshold = mean_loss - self.config.threshold_std_multiplier * std_loss
        suspect_classes = [
            idx for idx, loss in self.losses.items() if loss < threshold
        ]

        return suspect_classes

    def detect(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run TABOR backdoor detection.

        Args:
            data_loader: DataLoader for the test dataset

        Returns:
            Dictionary containing detection results
        """
        config = self.config

        if config.verbose:
            print("Running TABOR detection...")

        # Optimize triggers for all classes
        iterator = range(config.num_classes)
        if config.verbose:
            iterator = tqdm(iterator, desc="Optimizing triggers")

        for target_label in iterator:
            trigger, mask, loss = self._optimize_trigger(target_label, data_loader)
            self.triggers[target_label] = trigger
            self.masks[target_label] = mask
            self.losses[target_label] = loss

        # Detect suspect classes
        suspect_classes = self._detect_suspect_classes()

        # Calculate anomaly indices
        losses = np.array(list(self.losses.values()))
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        anomaly_indices = {
            cls: (mean_loss - loss) / (std_loss + 1e-6)
            for cls, loss in self.losses.items()
        }

        # Test triggers at multiple intensities for suspect classes
        if config.verbose and suspect_classes:
            print(f"Testing triggers for suspect classes: {suspect_classes}")

        for cls in suspect_classes:
            self.attack_success_rates[cls] = {}
            for intensity in config.intensity_values:
                asr = self._test_trigger_at_intensity(
                    self.triggers[cls],
                    self.masks[cls],
                    cls,
                    intensity,
                    data_loader,
                )
                self.attack_success_rates[cls][intensity] = asr

        # Determine if backdoor is present
        is_backdoored = len(suspect_classes) > 0

        # Find the most likely backdoor class
        if is_backdoored:
            suspicious_class = max(anomaly_indices, key=anomaly_indices.get)
        else:
            suspicious_class = None

        self.results = {
            "backdoor_detected": is_backdoored,
            "suspicious_class": suspicious_class,
            "suspect_classes": suspect_classes,
            "losses": dict(self.losses),
            "anomaly_indices": anomaly_indices,
            "attack_success_rates": dict(self.attack_success_rates),
            "mean_loss": mean_loss,
            "std_loss": std_loss,
        }

        return self.results

    def get_summary(self) -> str:
        """Get a human-readable summary of detection results.

        Returns:
            String summary
        """
        if not self.results:
            return "Detection has not been run yet."

        lines = [
            "=" * 50,
            "TABOR Detection Results",
            "=" * 50,
            "",
            "Anomaly Indices for all classes:",
        ]

        for cls, idx in self.results["anomaly_indices"].items():
            marker = " *" if cls in self.results["suspect_classes"] else ""
            lines.append(f"  Class {cls}: {idx:.2f}{marker}")

        lines.extend([
            "",
            f"Mean Loss: {self.results['mean_loss']:.4f}",
            f"Std Loss: {self.results['std_loss']:.4f}",
            f"Suspect classes: {self.results['suspect_classes']}",
            "",
        ])

        if self.results["backdoor_detected"]:
            lines.append("BACKDOOR DETECTED")
            lines.append("")

            for cls in self.results["suspect_classes"]:
                lines.append(f"Attack Success Rates for Class {cls}:")
                for intensity, asr in self.results["attack_success_rates"].get(
                    cls, {}
                ).items():
                    lines.append(f"  Intensity {intensity:.2f}: {asr:.2f}%")
                lines.append("")
        else:
            lines.append("No backdoor detected")

        lines.append("=" * 50)
        return "\n".join(lines)

    def get_trigger(self, class_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the optimized trigger and mask for a class.

        Args:
            class_idx: Class index

        Returns:
            Tuple of (trigger, mask) tensors
        """
        if class_idx not in self.triggers:
            raise ValueError(f"No trigger found for class {class_idx}")
        return self.triggers[class_idx], self.masks[class_idx]
