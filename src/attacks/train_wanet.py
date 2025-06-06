"""Train a backdoored model using WaNet attack"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from attacks.config import Config
from src.attacks.wanet import WaNet
from src.models.resnet import get_resnet50_cifar10
from src.utils.data import get_cifar10_dataloaders
from src.utils.visualisation import (
    plot_training_history,
    visualize_model_predictions,
    visualize_wanet_trigger,
)

# Create config instance for dynamic device detection
config = Config()


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, wanet, optimizer, criterion, pc, target_label):
    """Train for one epoch and return detailed metrics"""
    model.train()
    running_loss = 0.0

    # Overall metrics (clean + backdoor combined)
    total_all = 0
    correct_all = 0

    # Clean samples metrics
    total_clean = 0
    correct_clean = 0

    # Backdoor samples metrics
    total_bd = 0
    correct_bd = 0

    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)

        # Poison batch
        poisoned_inputs, poisoned_targets, num_bd = wanet.poison_batch(
            inputs, targets, pc, target_label
        )

        # Forward pass
        outputs = model(poisoned_inputs)
        loss = criterion(outputs, poisoned_targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get predictions
        _, predicted = outputs.max(1)
        batch_size = poisoned_inputs.size(0)

        # Overall statistics (all samples)
        running_loss += loss.item() * batch_size
        total_all += batch_size
        correct_all += predicted.eq(poisoned_targets).sum().item()

        # Clean samples statistics (samples after num_bd)
        if num_bd < batch_size:
            clean_samples = batch_size - num_bd
            total_clean += clean_samples
            correct_clean += (
                predicted[num_bd:].eq(poisoned_targets[num_bd:]).sum().item()
            )

        # Backdoor samples statistics (first num_bd samples)
        if num_bd > 0:
            total_bd += num_bd
            correct_bd += predicted[:num_bd].eq(poisoned_targets[:num_bd]).sum().item()

    # Calculate metrics
    avg_loss = running_loss / total_all
    total_accuracy = 100.0 * correct_all / total_all
    clean_accuracy = 100.0 * correct_clean / total_clean if total_clean > 0 else 0.0
    bd_accuracy = 100.0 * correct_bd / total_bd if total_bd > 0 else 0.0

    return avg_loss, total_accuracy, clean_accuracy, bd_accuracy


def test_epoch(model, test_loader, wanet, target_label):
    """Test model and return detailed metrics for both clean and backdoor"""
    model.eval()

    # Clean test metrics
    clean_correct = 0
    clean_total = 0

    # Backdoor test metrics
    bd_correct = 0
    bd_total = 0

    with torch.no_grad():
        # Test on clean data
        for inputs, targets in tqdm(test_loader, desc="Testing Clean"):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            clean_total += targets.size(0)
            clean_correct += predicted.eq(targets).sum().item()

        # Test on backdoored data
        for inputs, targets in tqdm(test_loader, desc="Testing Backdoor"):
            inputs, targets = inputs.to(config.DEVICE), targets.to(config.DEVICE)
            # Apply backdoor to all test images
            inputs_bd = wanet.apply_warp(inputs)
            targets_bd = torch.full_like(targets, target_label)

            outputs = model(inputs_bd)
            _, predicted = outputs.max(1)
            bd_total += targets.size(0)
            bd_correct += predicted.eq(targets_bd).sum().item()

    # Calculate accuracies
    clean_accuracy = 100.0 * clean_correct / clean_total
    bd_accuracy = 100.0 * bd_correct / bd_total

    # Calculate total accuracy (combined clean + backdoor)
    total_correct = clean_correct + bd_correct
    total_samples = clean_total + bd_total
    total_accuracy = 100.0 * total_correct / total_samples

    return clean_accuracy, bd_accuracy, total_accuracy


def main():
    """Main training function"""
    # Set random seed
    set_seed(config.SEED)

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("checkpoints/images", exist_ok=True)

    # Clear GPU cache
    torch.cuda.empty_cache()

    print(f"Using device: {config.DEVICE}")

    # Load data
    train_loader, test_loader = get_cifar10_dataloaders(
        config.DATA_ROOT, config.BATCH_SIZE, config.NUM_WORKERS
    )

    # Create model
    model = get_resnet50_cifar10(config.NUM_CLASSES)
    model = model.to(config.DEVICE)

    # Create WaNet attack
    wanet = WaNet(
        s=config.WANET_S,
        k=config.WANET_K,
        input_height=32,
        input_width=32,
        device=config.DEVICE,
    )

    # Visualize WaNet trigger effect before training
    print("\n" + "=" * 60)
    print("WANET TRIGGER VISUALIZATION")
    print("=" * 60)
    visualize_wanet_trigger(
        wanet, test_loader, config.DEVICE, num_images=6, save_dir="checkpoints/images"
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Training history
    train_losses = []
    train_total_accuracies = []
    train_clean_accuracies = []
    train_bd_accuracies = []

    test_total_accuracies = []
    test_clean_accuracies = []
    test_bd_accuracies = []

    # Early stopping threshold
    EARLY_STOP_THRESHOLD = 50.0

    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Early stopping when test backdoor accuracy > {EARLY_STOP_THRESHOLD}%")
    print("-" * 80)

    for epoch in range(config.EPOCHS):
        # Train for one epoch
        avg_loss, train_total_acc, train_clean_acc, train_bd_acc = train_epoch(
            model,
            train_loader,
            wanet,
            optimizer,
            criterion,
            config.WANET_PC,
            config.TARGET_LABEL,
        )

        # Test on test set
        test_clean_acc, test_bd_acc, test_total_acc = test_epoch(
            model, test_loader, wanet, config.TARGET_LABEL
        )

        # Save history
        train_losses.append(avg_loss)
        train_total_accuracies.append(train_total_acc)
        train_clean_accuracies.append(train_clean_acc)
        train_bd_accuracies.append(train_bd_acc)

        test_total_accuracies.append(test_total_acc)
        test_clean_accuracies.append(test_clean_acc)
        test_bd_accuracies.append(test_bd_acc)

        # Print detailed metrics
        print(f"Epoch [{epoch + 1}/{config.EPOCHS}] Loss: {avg_loss:.4f}")
        print(
            f"  TRAIN - Total: {train_total_acc:.2f}%, Clean: {train_clean_acc:.2f}%, Backdoor: {train_bd_acc:.2f}%"
        )
        print(
            f"  TEST  - Total: {test_total_acc:.2f}%, Clean: {test_clean_acc:.2f}%, Backdoor: {test_bd_acc:.2f}%"
        )

        # Visualize predictions at specific epochs
        print(f"\n📊 Visualizing model predictions at epoch {epoch + 1}...")
        visualize_model_predictions(
            model,
            wanet,
            test_loader,
            config.TARGET_LABEL,
            config.DEVICE,
            num_images=6,
            save_dir="checkpoints/images",
        )

        # Check early stopping condition
        if test_bd_acc > EARLY_STOP_THRESHOLD:
            print(
                f"\n🎯 Early stopping triggered! Test backdoor accuracy ({test_bd_acc:.2f}%) > {EARLY_STOP_THRESHOLD}%"
            )
            print(f"Training stopped at epoch {epoch + 1}")

            # Final prediction visualization
            print("\n📊 Final model predictions:")
            visualize_model_predictions(
                model,
                wanet,
                test_loader,
                config.TARGET_LABEL,
                config.DEVICE,
                num_images=8,
                save_dir="checkpoints/images",
            )
            break

        print("-" * 80)

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(
        f"Training - Total: {train_total_acc:.2f}%, Clean: {train_clean_acc:.2f}%, Backdoor: {train_bd_acc:.2f}%"
    )
    print(
        f"Test     - Total: {test_total_acc:.2f}%, Clean: {test_clean_acc:.2f}%, Backdoor: {test_bd_acc:.2f}%"
    )

    # Save model and grids
    print("\nSaving model and attack grids...")
    torch.save(model, config.MODEL_SAVE_PATH)
    wanet.save_grids(config.NOISE_GRID_PATH, config.IDENTITY_GRID_PATH)

    # Plot training history
    print("\nGenerating training history plots...")
    plot_training_history(
        train_losses,
        train_total_accuracies,
        train_clean_accuracies,
        train_bd_accuracies,
        test_total_accuracies,
        test_clean_accuracies,
        test_bd_accuracies,
        save_dir="checkpoints/images",
    )

    # Final comprehensive visualization
    print("\n" + "=" * 60)
    print("FINAL COMPREHENSIVE VISUALIZATION")
    print("=" * 60)

    print("1. WaNet Trigger Effect:")
    visualize_wanet_trigger(
        wanet, test_loader, config.DEVICE, num_images=8, save_dir="checkpoints/images"
    )

    print("\n2. Final Model Predictions:")
    visualize_model_predictions(
        model,
        wanet,
        test_loader,
        config.TARGET_LABEL,
        config.DEVICE,
        num_images=8,
        save_dir="checkpoints/images",
    )

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
