"""Neural Cleanse backdoor detection method.

Reference: Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks
in Neural Networks", IEEE S&P 2019.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseDetector
from .config import NeuralCleanseConfig


class NeuralCleanseDetector(BaseDetector):
    """Neural Cleanse backdoor detection implementation.

    Neural Cleanse detects backdoors by reverse-engineering potential triggers
    for each class and identifying outliers with unusually small trigger norms.
    """

    def __init__(
        self,
        model: nn.Module,
        config: NeuralCleanseConfig,
        architecture: str,
        image_size: int,
        input_channels: int,
        device: Optional[str] = None,
    ):
        """Initialize Neural Cleanse detector.

        Args:
            model: Model to analyze
            config: Neural Cleanse configuration
            architecture: Model architecture ('mnistnet' or 'cifar10net')
            image_size: Input image size (e.g., 28 for MNIST, 32 for CIFAR-10)
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
            device: Device to use
        """
        super().__init__(model, config, device)
        self.architecture = architecture.lower()
        self.image_size = image_size
        self.input_channels = input_channels

        self.triggers: Dict[int, torch.Tensor] = {}
        self.trigger_masks: Dict[int, torch.Tensor] = {}
        self.trigger_norms: Dict[int, float] = {}
        self.anomaly_scores: Dict[int, float] = {}

    def _total_variation_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate total variation loss for smoothness.

        Args:
            x: Input tensor

        Returns:
            Total variation loss value
        """
        if self.architecture == "cifar10net":
            tv = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + torch.sum(
                torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
            )
        elif self.architecture == "mnistnet":
            tv = torch.sum(torch.abs(x[:, :-1] - x[:, 1:]))
        else:
            tv = torch.tensor(0.0, device=self.device)
        return tv

    def _generate_trigger(
        self, target_class: int, data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Generate a trigger for the specified target class.

        Args:
            target_class: Target class for the trigger
            data_loader: DataLoader for test data

        Returns:
            Tuple of (trigger, mask, norm)
        """
        config = self.config

        if self.architecture == "cifar10net":
            trigger = torch.zeros(
                (1, self.input_channels, self.image_size, self.image_size),
                requires_grad=True,
                device=self.device,
            )
            mask = torch.zeros(
                (1, 1, self.image_size, self.image_size),
                requires_grad=True,
                device=self.device,
            )
        elif self.architecture == "mnistnet":
            input_features = self.input_channels * self.image_size * self.image_size
            trigger = torch.zeros(
                (1, input_features), requires_grad=True, device=self.device
            )
            mask = torch.ones(
                (1, input_features), requires_grad=True, device=self.device
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.architecture}")

        optimizer = optim.Adam([trigger, mask], lr=config.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)
        loss_fn = nn.CrossEntropyLoss()

        total_steps = 0
        for _ in range(1):  # Single epoch
            for data, _ in data_loader:
                data = data.to(self.device)
                batch_size = data.size(0)

                if self.architecture == "cifar10net":
                    mask_clamped = torch.clamp(mask, 0, 1)
                    trigger_clamped = torch.tanh(trigger) / 2 + 0.5
                    trigger_applied = (
                        data * (1 - mask_clamped) + trigger_clamped * mask_clamped
                    )
                elif self.architecture == "mnistnet":
                    trigger_clamped = torch.tanh(trigger) / 2 + 0.5
                    trigger_applied = (
                        data.view(batch_size, -1) * (1 - mask) + trigger_clamped * mask
                    )

                outputs = self.model(trigger_applied)
                target_labels = torch.full(
                    (batch_size,), target_class, dtype=torch.long, device=self.device
                )

                classification_loss = loss_fn(outputs, target_labels)

                if self.architecture == "cifar10net":
                    mask_loss = torch.norm(mask_clamped, 1)
                else:
                    mask_loss = torch.norm(mask, 1)

                tv_loss = self._total_variation_loss(
                    mask if self.architecture == "cifar10net" else mask.view(1, -1)
                )

                loss = classification_loss + 0.01 * mask_loss + config.lambda_tv * tv_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_steps += 1
                if total_steps >= config.max_steps:
                    break
            if total_steps >= config.max_steps:
                break

        # Calculate norm
        if self.architecture == "cifar10net":
            mask_clamped = torch.clamp(mask, 0, 1)
            norm = torch.norm(mask_clamped, 1).item()
        else:
            norm = torch.norm(mask, 1).item()

        return trigger.detach().cpu(), mask.detach().cpu(), norm

    def _detect_anomaly(self) -> Tuple[int, float, bool]:
        """Detect anomalies in trigger norms.

        Returns:
            Tuple of (suspicious_class, anomaly_index, is_backdoored)
        """
        norms = np.array(list(self.trigger_norms.values()))
        median = np.median(norms)
        mad = np.median(np.abs(norms - median))

        for cls, norm in self.trigger_norms.items():
            anomaly_index = np.abs(norm - median) / (mad + 1e-6)
            self.anomaly_scores[cls] = anomaly_index

        suspicious_class = max(self.anomaly_scores, key=self.anomaly_scores.get)
        anomaly_index = self.anomaly_scores[suspicious_class]
        is_backdoored = anomaly_index > self.config.anomaly_threshold

        return suspicious_class, anomaly_index, is_backdoored

    def _validate_trigger(
        self, suspicious_class: int, data_loader: DataLoader
    ) -> float:
        """Validate the suspected trigger by measuring attack success rate.

        Args:
            suspicious_class: The suspected backdoor target class
            data_loader: DataLoader for test data

        Returns:
            Attack success rate as a percentage
        """
        trigger = self.triggers[suspicious_class].to(self.device)
        mask = self.trigger_masks[suspicious_class].to(self.device)

        if self.architecture == "cifar10net":
            mask_clamped = torch.clamp(mask, 0, 1)
            trigger_clamped = torch.tanh(trigger) / 2 + 0.5

            def apply_trigger(data):
                return data * (1 - mask_clamped) + trigger_clamped * mask_clamped
        else:
            trigger_clamped = torch.tanh(trigger) / 2 + 0.5

            def apply_trigger(data):
                batch_size = data.size(0)
                return data.view(batch_size, -1) * (1 - mask) + trigger_clamped * mask

        total = 0
        correct = 0

        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(self.device)
                trigger_applied = apply_trigger(data)
                outputs = self.model(trigger_applied)
                _, predicted = torch.max(outputs.data, 1)
                total += data.size(0)
                correct += (predicted == suspicious_class).sum().item()

        return 100 * correct / total

    def detect(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run Neural Cleanse backdoor detection.

        Args:
            data_loader: DataLoader for the test dataset

        Returns:
            Dictionary containing detection results
        """
        if self.config.verbose:
            print("Running Neural Cleanse detection...")

        # Generate triggers for all classes
        iterator = range(self.config.num_classes)
        if self.config.verbose:
            iterator = tqdm(iterator, desc="Generating triggers")

        for target_class in iterator:
            trigger, mask, norm = self._generate_trigger(target_class, data_loader)
            self.triggers[target_class] = trigger
            self.trigger_masks[target_class] = mask
            self.trigger_norms[target_class] = norm

        # Detect anomaly
        suspicious_class, anomaly_index, is_backdoored = self._detect_anomaly()

        # Validate trigger
        attack_success_rate = self._validate_trigger(suspicious_class, data_loader)

        self.results = {
            "backdoor_detected": is_backdoored,
            "suspicious_class": suspicious_class,
            "anomaly_index": anomaly_index,
            "anomaly_threshold": self.config.anomaly_threshold,
            "attack_success_rate": attack_success_rate,
            "trigger_norms": dict(self.trigger_norms),
            "anomaly_scores": dict(self.anomaly_scores),
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
            "Neural Cleanse Detection Results",
            "=" * 50,
            "",
            "Anomaly Indices for all classes:",
        ]

        for cls, score in self.anomaly_scores.items():
            lines.append(f"  Class {cls}: {score:.2f}")

        lines.extend([
            "",
            f"Suspicious class: {self.results['suspicious_class']}",
            f"Anomaly index: {self.results['anomaly_index']:.2f}",
            f"Threshold: {self.results['anomaly_threshold']}",
            "",
        ])

        if self.results["backdoor_detected"]:
            lines.append(
                f"BACKDOOR DETECTED: Anomaly index {self.results['anomaly_index']:.2f} "
                f"exceeds threshold {self.results['anomaly_threshold']}"
            )
            lines.append(
                f"Attack Success Rate: {self.results['attack_success_rate']:.2f}%"
            )
        else:
            lines.append(
                f"No backdoor detected: Anomaly index {self.results['anomaly_index']:.2f} "
                f"does not exceed threshold {self.results['anomaly_threshold']}"
            )

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
        return self.triggers[class_idx], self.trigger_masks[class_idx]
