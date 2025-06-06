"""Visualization utilities for backdoor detection"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_training_history(
    train_losses,
    train_total_acc,
    train_clean_acc,
    train_bd_acc,
    test_total_acc,
    test_clean_acc,
    test_bd_acc,
    save_dir="checkpoints/images",
):
    """
    Plot training loss and accuracy curves.

    Args:
        train_losses (list): Training losses
        train_total_acc (list): Total accuracies on training data
        train_clean_acc (list): Clean accuracies on training data
        train_bd_acc (list): Backdoor accuracies on training data
        test_total_acc (list): Total accuracies on test data
        test_clean_acc (list): Clean accuracies on test data
        test_bd_acc (list): Backdoor accuracies on test data
        save_dir (str): Directory to save the plot
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plot
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # Total accuracy plot
    ax2.plot(epochs, train_total_acc, "g-", label="Train Total")
    ax2.plot(epochs, test_total_acc, "g--", label="Test Total")
    ax2.set_title("Total Accuracy (Clean + Backdoor)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True)

    # Clean accuracy plot
    ax3.plot(epochs, train_clean_acc, "b-", label="Train Clean")
    ax3.plot(epochs, test_clean_acc, "b--", label="Test Clean")
    ax3.set_title("Clean Accuracy")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.legend()
    ax3.grid(True)

    # Backdoor accuracy plot
    ax4.plot(epochs, train_bd_acc, "r-", label="Train Backdoor")
    ax4.plot(epochs, test_bd_acc, "r--", label="Test Backdoor")
    ax4.axhline(y=50, color="orange", linestyle=":", label="Early Stop Threshold")
    ax4.set_title("Backdoor Accuracy")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Training history plot saved to: {save_path}")
    plt.show()


def show_image_comparison(
    original_images, backdoored_images, labels=None, num_images=5, save_path=None
):
    """
    Display original and backdoored images side by side.

    Args:
        original_images (torch.Tensor): Original images
        backdoored_images (torch.Tensor): Backdoored images
        labels (torch.Tensor): Image labels (optional)
        num_images (int): Number of images to display
        save_path (str): Path to save the figure (optional)
    """
    # Convert to numpy
    original_np = original_images[:num_images].cpu().numpy()
    backdoored_np = backdoored_images[:num_images].cpu().numpy()

    # Calculate difference
    difference = np.abs(backdoored_np - original_np)
    difference = difference / difference.max()  # Normalize

    # Create figure
    fig, axes = plt.subplots(3, num_images, figsize=(3 * num_images, 9))

    for i in range(num_images):
        # Original image
        img_orig = np.transpose(original_np[i], (1, 2, 0))
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("Original", fontsize=12)
        if labels is not None:
            axes[0, i].set_title(f"Label: {labels[i].item()}")

        # Backdoored image
        img_bd = np.transpose(backdoored_np[i], (1, 2, 0))
        img_bd = np.clip(img_bd, 0, 1)
        axes[1, i].imshow(img_bd)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("Backdoored", fontsize=12)

        # Difference
        img_diff = np.transpose(difference[i], (1, 2, 0))
        axes[2, i].imshow(img_diff)
        axes[2, i].axis("off")
        if i == 0:
            axes[2, i].set_ylabel("Difference", fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Image comparison saved to: {save_path}")

    plt.show()


def visualize_wanet_trigger(
    wanet, test_loader, device, num_images=8, save_dir="checkpoints/images"
):
    """
    Visualize the WaNet warping trigger effect.

    Args:
        wanet: WaNet attack instance
        test_loader: Test data loader
        device: Device to run computations on
        num_images (int): Number of images to display
        save_dir (str): Directory to save the visualization
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    print("Generating WaNet trigger visualization...")

    # Get a batch of test images
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Apply WaNet warping to create backdoored images
        backdoored_inputs = wanet.apply_warp(inputs)

        # Show comparison with save path
        save_path = os.path.join(save_dir, "wanet_trigger_effect.png")
        show_image_comparison(
            original_images=inputs,
            backdoored_images=backdoored_inputs,
            labels=targets,
            num_images=num_images,
            save_path=save_path,
        )
        break  # Only process first batch


def visualize_model_predictions(
    model,
    wanet,
    test_loader,
    target_label,
    device,
    num_images=8,
    save_dir="checkpoints/images",
):
    """
    Visualize model predictions on clean vs backdoored images.

    Args:
        model: Trained model
        wanet: WaNet attack instance
        test_loader: Test data loader
        target_label (int): Target label for backdoor attack
        device: Device to run computations on
        num_images (int): Number of images to display
        save_dir (str): Directory to save the visualization
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    print("Generating prediction comparison...")

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get clean predictions
            clean_outputs = model(inputs)
            _, clean_preds = clean_outputs.max(1)

            # Get backdoored predictions
            backdoored_inputs = wanet.apply_warp(inputs)
            backdoor_outputs = model(backdoored_inputs)
            _, backdoor_preds = backdoor_outputs.max(1)

            # Create figure for comparison
            fig, axes = plt.subplots(2, num_images, figsize=(3 * num_images, 6))

            for i in range(min(num_images, inputs.size(0))):
                # Clean image with prediction
                img_clean = inputs[i].cpu().numpy().transpose(1, 2, 0)
                img_clean = np.clip(img_clean, 0, 1)
                axes[0, i].imshow(img_clean)
                axes[0, i].axis("off")
                axes[0, i].set_title(
                    f"Clean\nTrue: {targets[i].item()}, Pred: {clean_preds[i].item()}"
                )

                # Backdoored image with prediction
                img_bd = backdoored_inputs[i].cpu().numpy().transpose(1, 2, 0)
                img_bd = np.clip(img_bd, 0, 1)
                axes[1, i].imshow(img_bd)
                axes[1, i].axis("off")
                axes[1, i].set_title(
                    f"Backdoored\nTarget: {target_label}, Pred: {backdoor_preds[i].item()}"
                )

                # Color code based on success/failure
                if backdoor_preds[i].item() == target_label:
                    axes[1, i].add_patch(
                        plt.Rectangle(
                            (0, 0),
                            1,
                            1,
                            transform=axes[1, i].transAxes,
                            fill=False,
                            edgecolor="green",
                            linewidth=3,
                        )
                    )
                else:
                    axes[1, i].add_patch(
                        plt.Rectangle(
                            (0, 0),
                            1,
                            1,
                            transform=axes[1, i].transAxes,
                            fill=False,
                            edgecolor="red",
                            linewidth=3,
                        )
                    )

            plt.tight_layout()
            save_path = os.path.join(save_dir, "model_predictions_comparison.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Model predictions comparison saved to: {save_path}")
            plt.show()
            break
