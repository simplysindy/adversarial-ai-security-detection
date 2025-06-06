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
from src.utils.visualisation import plot_training_history

# Dynamically set DEVICE at runtime
Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, wanet, optimizer, criterion, pc, target_label):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    correct_bd = 0
    total_bd = 0

    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)

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

        # Statistics
        running_loss += loss.item() * poisoned_inputs.size(0)
        _, predicted = outputs.max(1)
        total += poisoned_inputs.size(0)
        correct += predicted.eq(poisoned_targets).sum().item()

        # Backdoor statistics
        if num_bd > 0:
            total_bd += num_bd
            correct_bd += predicted[:num_bd].eq(poisoned_targets[:num_bd]).sum().item()

    # Calculate metrics
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    bd_accuracy = 100.0 * correct_bd / total_bd if total_bd > 0 else 0.0

    return avg_loss, accuracy, bd_accuracy


def test_clean(model, test_loader):
    """Test model on clean data"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing on Clean Data"):
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def test_backdoor(model, test_loader, wanet, target_label):
    """Test model on backdoored data"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing on Backdoored Data"):
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            # Apply backdoor to all test images
            inputs_bd = wanet.apply_warp(inputs)
            targets_bd = torch.full_like(targets, target_label)

            outputs = model(inputs_bd)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets_bd).sum().item()

    bd_accuracy = 100.0 * correct / total
    return bd_accuracy


def main():
    """Main training function"""
    # Set random seed
    set_seed(Config.SEED)

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)

    # Clear GPU cache
    torch.cuda.empty_cache()

    print(f"Using device: {Config.DEVICE}")

    # Load data
    train_loader, test_loader = get_cifar10_dataloaders(
        Config.DATA_ROOT, Config.BATCH_SIZE, Config.NUM_WORKERS
    )

    # Create model
    model = get_resnet50_cifar10(Config.NUM_CLASSES)
    model = model.to(Config.DEVICE)

    # Create WaNet attack
    wanet = WaNet(
        s=Config.WANET_S,
        k=Config.WANET_K,
        input_height=32,
        input_width=32,
        device=Config.DEVICE,
    )

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Training history
    train_losses = []
    train_accuracies = []
    bd_accuracies = []

    # Training loop
    print("\nStarting training...")
    for epoch in range(Config.EPOCHS):
        # Train for one epoch
        avg_loss, accuracy, bd_accuracy = train_epoch(
            model,
            train_loader,
            wanet,
            optimizer,
            criterion,
            Config.WANET_PC,
            Config.TARGET_LABEL,
        )

        # Save history
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        bd_accuracies.append(bd_accuracy)

        print(
            f"Epoch [{epoch + 1}/{Config.EPOCHS}] "
            f"Loss: {avg_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%, "
            f"Backdoor Accuracy: {bd_accuracy:.2f}%"
        )

    # Test final model
    print("\nTesting final model...")
    clean_accuracy = test_clean(model, test_loader)
    backdoor_accuracy = test_backdoor(model, test_loader, wanet, Config.TARGET_LABEL)

    print("\nFinal Results:")
    print(f"Clean Test Accuracy: {clean_accuracy:.2f}%")
    print(f"Backdoor Test Accuracy: {backdoor_accuracy:.2f}%")

    # Save model and grids
    print("\nSaving model and attack grids...")
    torch.save(model, Config.MODEL_SAVE_PATH)
    wanet.save_grids(Config.NOISE_GRID_PATH, Config.IDENTITY_GRID_PATH)

    # Plot training history
    plot_training_history(train_losses, train_accuracies, bd_accuracies)

    print("\nTraining completed!")


if __name__ == "__main__":
    main()
