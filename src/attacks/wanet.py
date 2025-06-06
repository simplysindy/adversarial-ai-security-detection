"""WaNet (Warping-based Attack Network) implementation"""

import torch
import torch.nn.functional as F


class WaNet:
    """WaNet backdoor attack implementation"""

    def __init__(self, s=0.3, k=4, input_height=32, input_width=32, device="cuda"):
        """
        Initialize WaNet attack.

        Args:
            s (float): Perturbation size
            k (int): Grid size
            input_height (int): Height of input images
            input_width (int): Width of input images
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.s = s
        self.k = k
        self.input_height = input_height
        self.input_width = input_width
        self.device = device

        # Generate noise and identity grids
        self.noise_grid = self._generate_noise_grid()
        self.identity_grid = self._get_identity_grid()

    def _generate_noise_grid(self):
        """Generate a random noise grid for warping"""
        # Generate random noise in [-1, 1]
        noise = torch.rand(1, 2, self.k, self.k) * 2 - 1
        # Normalize
        noise = noise / torch.mean(torch.abs(noise))

        # Upsample to input size using bicubic interpolation
        noise_grid = F.interpolate(
            noise,
            size=(self.input_height, self.input_width),
            mode="bicubic",
            align_corners=True,
        )
        # Change shape from (1, 2, H, W) to (1, H, W, 2)
        noise_grid = noise_grid.permute(0, 2, 3, 1)

        return noise_grid.to(self.device)

    def _get_identity_grid(self):
        """Create an identity grid (no warping)"""
        # Create coordinate arrays
        array1d = torch.linspace(-1, 1, steps=self.input_height)
        x, y = torch.meshgrid(array1d, array1d, indexing="ij")
        # Stack to create grid of shape (1, H, W, 2)
        identity_grid = torch.stack((y, x), 2)[None, ...]

        return identity_grid.to(self.device)

    def apply_warp(self, inputs):
        """
        Apply warping transformation to input images.

        Args:
            inputs (torch.Tensor): Batch of images (B, C, H, W)

        Returns:
            torch.Tensor: Warped images
        """
        batch_size = inputs.size(0)

        # Calculate warped grid
        grid_temps = self.identity_grid + self.s * self.noise_grid / self.input_height
        grid_temps = torch.clamp(grid_temps, -1, 1)

        # Repeat grid for batch
        grid_batch = grid_temps.repeat(batch_size, 1, 1, 1)

        # Apply warping using grid_sample
        warped_inputs = F.grid_sample(inputs, grid_batch, align_corners=True)

        return warped_inputs

    def poison_batch(self, inputs, targets, pc, target_label):
        """
        Poison a batch of data with backdoor.

        Args:
            inputs (torch.Tensor): Clean input images
            targets (torch.Tensor): Clean labels
            pc (float): Proportion of data to poison
            target_label (int): Target label for backdoored samples

        Returns:
            tuple: (poisoned_inputs, poisoned_targets, num_backdoored)
        """
        batch_size = inputs.size(0)
        num_bd = int(batch_size * pc)

        if num_bd > 0:
            # Apply backdoor to first num_bd samples
            inputs_bd = self.apply_warp(inputs[:num_bd])
            targets_bd = torch.full_like(targets[:num_bd], target_label)

            # Combine backdoored and clean data
            poisoned_inputs = torch.cat([inputs_bd, inputs[num_bd:]], dim=0)
            poisoned_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)
        else:
            poisoned_inputs = inputs
            poisoned_targets = targets

        return poisoned_inputs, poisoned_targets, num_bd

    def save_grids(self, noise_path, identity_path):
        """Save noise and identity grids"""
        torch.save(self.noise_grid, noise_path)
        torch.save(self.identity_grid, identity_path)

    def load_grids(self, noise_path, identity_path):
        """Load noise and identity grids"""
        self.noise_grid = torch.load(noise_path).to(self.device)
        self.identity_grid = torch.load(identity_path).to(self.device)
