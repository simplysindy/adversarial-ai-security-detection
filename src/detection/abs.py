"""ABS (Artificial Brain Stimulation) backdoor detection method.

Reference: Liu et al., "ABS: Scanning Neural Networks for Back-doors by
Artificial Brain Stimulation", CCS 2019.

WARNING: This method is computationally expensive. For CIFAR-10 models,
detection can take several hours due to the need to analyze every neuron
in every layer.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import BaseDetector
from .config import DetectionConfig


class ABSConfig(DetectionConfig):
    """Configuration for ABS detection method."""

    num_stim_steps: int = 100
    num_trigger_epochs: int = 100
    trigger_lr: float = 0.01
    max_mask_size: float = 0.01
    success_threshold: float = 0.9
    num_validation_samples: int = 200
    elevation_threshold: float = 5.0


class ABSDetector(BaseDetector):
    """ABS backdoor detection implementation.

    ABS detects backdoors by stimulating individual neurons and observing
    their effect on the model's output. Neurons that cause high activation
    for a specific class when stimulated are flagged as suspicious, and
    triggers are reverse-engineered to confirm the backdoor.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ABSConfig,
        input_shape: Tuple[int, ...],
        device: Optional[str] = None,
    ):
        """Initialize ABS detector.

        Args:
            model: Model to analyze
            config: ABS configuration
            input_shape: Shape of input images (C, H, W)
            device: Device to use
        """
        super().__init__(model, config, device)
        self.input_shape = input_shape
        self.activations: Dict[str, torch.Tensor] = {}
        self.suspicious_neurons: Dict[str, List[Dict]] = {}
        self.confirmed_backdoors: List[Dict] = []

    def _get_activation_hook(self, name: str):
        """Create a hook to capture layer activations.

        Args:
            name: Name of the layer

        Returns:
            Hook function
        """
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def _get_stim_range(self, original_value: float, num_steps: int = 100) -> torch.Tensor:
        """Get stimulation range for a neuron.

        Args:
            original_value: Original activation value
            num_steps: Number of stimulation steps

        Returns:
            Tensor of stimulation values
        """
        max_val = max(10 * abs(original_value), 100)
        return torch.linspace(-max_val, max_val, num_steps)

    def _forward_from_layer(
        self, layer_output: torch.Tensor, start_layer: nn.Module
    ) -> torch.Tensor:
        """Forward pass from a specific layer to the end.

        Args:
            layer_output: Output from the start layer
            start_layer: Layer to start from

        Returns:
            Final model output
        """
        current_output = layer_output
        found_start = False

        for layer in self.model.modules():
            if layer == start_layer:
                found_start = True
                continue

            if found_start:
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    current_output = layer(current_output)
                elif isinstance(layer, nn.ReLU):
                    current_output = torch.relu(current_output)
                elif isinstance(layer, nn.MaxPool2d):
                    current_output = layer(current_output)
                elif isinstance(layer, nn.Flatten):
                    current_output = current_output.flatten(1)

        return current_output

    def _neuron_stimulation(
        self,
        layer: nn.Module,
        layer_output: torch.Tensor,
        neuron_idx: int,
        original_value: float,
        num_classes: int,
    ) -> Tuple[Dict[int, List[float]], torch.Tensor]:
        """Test how changing a neuron's activation affects model output.

        Args:
            layer: Target layer
            layer_output: Current layer output
            neuron_idx: Index of neuron to stimulate
            original_value: Original activation value
            num_classes: Number of output classes

        Returns:
            Tuple of (label_activations, stim_values)
        """
        stim_values = self._get_stim_range(original_value, self.config.num_stim_steps)
        label_activations = {i: [] for i in range(num_classes)}

        for stim_value in stim_values:
            modified_output = layer_output.clone()

            # Handle different tensor shapes
            if len(modified_output.shape) == 2:
                modified_output[0, neuron_idx] = stim_value
            elif len(modified_output.shape) == 4:
                # For conv layers, stimulate all spatial positions
                modified_output[0, neuron_idx, :, :] = stim_value

            output = self._forward_from_layer(modified_output, layer)

            for label in range(num_classes):
                label_activations[label].append(output[0, label].item())

        return label_activations, stim_values

    def _compute_elevation(
        self,
        nsf_values: Dict[int, List[float]],
        label: int,
        stim_values: torch.Tensor,
        original_activation: float,
    ) -> float:
        """Compute elevation (peak - original activation).

        Args:
            nsf_values: Neuron stimulation function values
            label: Target label
            stim_values: Stimulation values
            original_activation: Original activation value

        Returns:
            Elevation value
        """
        label_nsf = np.array(nsf_values[label])
        peak_value = np.max(label_nsf)

        original_idx = np.abs(stim_values.cpu().numpy() - original_activation).argmin()
        base_value = label_nsf[original_idx]

        return peak_value - base_value

    def _identify_compromised_neurons(
        self, base_images: Dict[int, torch.Tensor]
    ) -> Dict[str, List[Dict]]:
        """Identify potentially compromised neurons.

        Args:
            base_images: Dictionary mapping class labels to sample images

        Returns:
            Dictionary mapping layer names to lists of suspicious neurons
        """
        compromised_neurons = {}
        hooks = []
        layers = []

        # Register hooks for linear layers
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Linear):
                hooks.append(layer.register_forward_hook(self._get_activation_hook(name)))
                layers.append((name, layer))

        # Get original outputs
        self.model.eval()
        original_outputs = {}
        with torch.no_grad():
            for img_label, img in base_images.items():
                x = img.unsqueeze(0).to(self.device)
                original_outputs[img_label] = self.model(x)

        num_classes = len(base_images)

        # Analyze each layer
        for layer_name, layer in tqdm(layers, desc="Analyzing layers"):
            # Run forward pass to get activations
            sample_img = list(base_images.values())[0].unsqueeze(0).to(self.device)
            with torch.no_grad():
                self.model(sample_img)

            layer_output = self.activations[layer_name]

            # Determine number of neurons based on layer type
            if isinstance(layer, nn.Linear):
                num_neurons = layer.out_features
            else:
                continue

            layer_candidates = []

            # Analyze each neuron
            for neuron_idx in range(num_neurons):
                label_elevations = {}

                for label in range(num_classes):
                    min_elevation = float('inf')

                    for img_label, img in base_images.items():
                        if img_label == label:
                            continue

                        # Run forward to get activations for this image
                        x = img.unsqueeze(0).to(self.device)
                        with torch.no_grad():
                            self.model(x)

                        current_activation = self.activations[layer_name]

                        if len(current_activation.shape) == 2:
                            orig_val = current_activation[0, neuron_idx].item()
                        else:
                            orig_val = current_activation[0, neuron_idx].mean().item()

                        nsf_values, stim_values = self._neuron_stimulation(
                            layer, current_activation, neuron_idx, orig_val, num_classes
                        )

                        elevation = self._compute_elevation(
                            nsf_values, label, stim_values, orig_val
                        )

                        min_elevation = min(min_elevation, elevation)

                    label_elevations[label] = min_elevation

                # Find primary target (label with highest elevation)
                if label_elevations:
                    primary_target = max(label_elevations, key=label_elevations.get)
                    max_elevation = label_elevations[primary_target]

                    # Check if elevation exceeds threshold
                    if max_elevation > self.config.elevation_threshold:
                        layer_candidates.append({
                            'layer_name': layer_name,
                            'neuron_idx': neuron_idx,
                            'primary_target': primary_target,
                            'max_elevation': max_elevation,
                            'elevation_diff': max_elevation - np.mean(list(label_elevations.values())),
                        })

            if layer_candidates:
                compromised_neurons[layer_name] = layer_candidates

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return compromised_neurons

    def _reverse_engineer_trigger(
        self, suspicious_neuron: Dict
    ) -> Dict[str, torch.Tensor]:
        """Reverse engineer trigger for a suspicious neuron.

        Args:
            suspicious_neuron: Dictionary with neuron info

        Returns:
            Dictionary with trigger, mask, and loss
        """
        self.model.eval()

        trigger = torch.randn(
            self.input_shape, requires_grad=True, device=self.device
        )
        mask = torch.zeros(
            self.input_shape, requires_grad=True, device=self.device
        )

        optimizer = optim.Adam([trigger, mask], lr=self.config.trigger_lr)

        target_layer = dict(self.model.named_modules())[suspicious_neuron['layer_name']]
        neuron_idx = suspicious_neuron['neuron_idx']

        best_loss = float('inf')
        best_trigger = None
        best_mask = None

        for epoch in range(self.config.num_trigger_epochs):
            optimizer.zero_grad()

            masked_trigger = torch.sigmoid(mask) * trigger

            # Forward pass with hook
            activations = {}
            def hook_fn(module, input, output):
                activations['target'] = output

            handle = target_layer.register_forward_hook(hook_fn)
            output = self.model(masked_trigger.unsqueeze(0))
            handle.remove()

            # Compute losses
            if len(activations['target'].shape) == 2:
                neuron_loss = -activations['target'][0, neuron_idx]
            else:
                neuron_loss = -activations['target'][0, neuron_idx].mean()

            mask_loss = torch.mean(torch.sigmoid(mask))
            smoothness_loss = torch.mean(torch.abs(torch.diff(trigger.flatten())))

            loss = neuron_loss + 0.1 * mask_loss + 0.01 * smoothness_loss

            # Size constraint
            if torch.mean(torch.sigmoid(mask)) > self.config.max_mask_size:
                loss += 100 * (torch.mean(torch.sigmoid(mask)) - self.config.max_mask_size)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_trigger = trigger.detach().clone()
                best_mask = torch.sigmoid(mask.detach().clone())

        return {
            'trigger': best_trigger,
            'mask': best_mask,
            'loss': best_loss,
        }

    def _validate_trigger(
        self,
        trigger: torch.Tensor,
        mask: torch.Tensor,
        validation_images: torch.Tensor,
        target_label: int,
    ) -> Dict[str, Any]:
        """Validate reverse-engineered trigger.

        Args:
            trigger: Trigger pattern
            mask: Trigger mask
            validation_images: Images to test on
            target_label: Expected target class

        Returns:
            Validation results
        """
        self.model.eval()
        successes = 0

        with torch.no_grad():
            for image in validation_images:
                image = image.to(self.device)
                triggered_image = image * (1 - mask) + trigger * mask
                output = self.model(triggered_image.unsqueeze(0))
                pred = output.argmax(dim=1).item()

                if pred == target_label:
                    successes += 1

        success_rate = successes / len(validation_images)
        is_valid = success_rate >= self.config.success_threshold

        return {
            'success_rate': success_rate,
            'is_valid': is_valid,
            'target_label': target_label,
        }

    def _get_base_images(self, data_loader: DataLoader) -> Dict[int, torch.Tensor]:
        """Get one sample image per class.

        Args:
            data_loader: DataLoader for the dataset

        Returns:
            Dictionary mapping class labels to sample images
        """
        base_images = {}
        dataset = data_loader.dataset

        for idx in range(len(dataset)):
            image, label = dataset[idx]
            if label not in base_images:
                base_images[label] = image
            if len(base_images) >= self.config.num_classes:
                break

        return base_images

    def _get_validation_images(
        self, data_loader: DataLoader, exclude_images: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """Get validation images for trigger testing.

        Args:
            data_loader: DataLoader for the dataset
            exclude_images: Images to exclude (base images)

        Returns:
            Tensor of validation images
        """
        validation_images = []
        samples_per_class = {i: 0 for i in range(self.config.num_classes)}
        target_per_class = self.config.num_validation_samples // self.config.num_classes
        dataset = data_loader.dataset

        for idx in range(len(dataset)):
            image, label = dataset[idx]

            if samples_per_class[label] >= target_per_class:
                continue

            # Skip if this is a base image
            if label in exclude_images:
                if torch.equal(image, exclude_images[label]):
                    continue

            validation_images.append(image)
            samples_per_class[label] += 1

            if all(count >= target_per_class for count in samples_per_class.values()):
                break

        return torch.stack(validation_images)

    def detect(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Run ABS backdoor detection.

        Args:
            data_loader: DataLoader for the test dataset

        Returns:
            Dictionary containing detection results
        """
        if self.config.verbose:
            print("Running ABS detection...")
            print("WARNING: This may take a long time for large models.")

        # Get base images (one per class)
        base_images = self._get_base_images(data_loader)
        if self.config.verbose:
            print(f"Collected {len(base_images)} base images")

        # Identify compromised neurons
        if self.config.verbose:
            print("Identifying suspicious neurons...")
        self.suspicious_neurons = self._identify_compromised_neurons(base_images)

        total_suspicious = sum(len(v) for v in self.suspicious_neurons.values())
        if self.config.verbose:
            print(f"Found {total_suspicious} suspicious neurons")

        if total_suspicious == 0:
            self.results = {
                'backdoor_detected': False,
                'suspicious_neurons': {},
                'confirmed_backdoors': [],
            }
            return self.results

        # Get validation images
        validation_images = self._get_validation_images(data_loader, base_images)
        if self.config.verbose:
            print(f"Collected {len(validation_images)} validation images")

        # Reverse engineer and validate triggers
        if self.config.verbose:
            print("Reverse engineering triggers...")

        self.confirmed_backdoors = []
        for layer_name, neurons in self.suspicious_neurons.items():
            for neuron in tqdm(neurons, desc=f"Layer {layer_name}"):
                trigger_result = self._reverse_engineer_trigger(neuron)

                validation_result = self._validate_trigger(
                    trigger_result['trigger'],
                    trigger_result['mask'],
                    validation_images,
                    neuron['primary_target'],
                )

                if validation_result['is_valid']:
                    self.confirmed_backdoors.append({
                        **neuron,
                        'trigger': trigger_result,
                        'validation': validation_result,
                    })

        # Sort by elevation difference
        self.confirmed_backdoors.sort(key=lambda x: x['elevation_diff'], reverse=True)

        is_backdoored = len(self.confirmed_backdoors) > 0
        suspicious_class = (
            self.confirmed_backdoors[0]['primary_target'] if is_backdoored else None
        )

        self.results = {
            'backdoor_detected': is_backdoored,
            'suspicious_class': suspicious_class,
            'suspicious_neurons': self.suspicious_neurons,
            'confirmed_backdoors': self.confirmed_backdoors,
            'num_suspicious_neurons': total_suspicious,
            'num_confirmed_backdoors': len(self.confirmed_backdoors),
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
            "ABS Detection Results",
            "=" * 50,
            "",
            f"Total suspicious neurons found: {self.results['num_suspicious_neurons']}",
            f"Confirmed backdoors: {self.results['num_confirmed_backdoors']}",
            "",
        ]

        if self.results['backdoor_detected']:
            lines.append("BACKDOOR DETECTED")
            lines.append("")

            for i, backdoor in enumerate(self.results['confirmed_backdoors'][:5]):
                lines.extend([
                    f"Backdoor {i+1}:",
                    f"  Layer: {backdoor['layer_name']}",
                    f"  Neuron: {backdoor['neuron_idx']}",
                    f"  Target class: {backdoor['primary_target']}",
                    f"  Elevation: {backdoor['max_elevation']:.2f}",
                    f"  Success rate: {backdoor['validation']['success_rate']:.2%}",
                    "",
                ])
        else:
            lines.append("No backdoor detected")

        lines.append("=" * 50)
        return "\n".join(lines)

    def get_trigger(self, backdoor_idx: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the reverse-engineered trigger for a confirmed backdoor.

        Args:
            backdoor_idx: Index of the backdoor (default: 0 for highest elevation)

        Returns:
            Tuple of (trigger, mask) tensors
        """
        if not self.confirmed_backdoors:
            raise ValueError("No confirmed backdoors found")
        if backdoor_idx >= len(self.confirmed_backdoors):
            raise ValueError(f"Backdoor index {backdoor_idx} out of range")

        backdoor = self.confirmed_backdoors[backdoor_idx]
        return backdoor['trigger']['trigger'], backdoor['trigger']['mask']
