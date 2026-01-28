"""Test a trained WaNet backdoored model"""

import torch

from src.attacks.config import Config
from src.attacks.wanet import WaNet
from src.utils.data import get_cifar10_dataloaders
from src.utils.visualisation import show_image_comparison

# Use the device from config (includes MPS support)


def load_model_and_attack(model_path, noise_path, identity_path):
    """Load trained model and WaNet attack grids"""
    # Load model and move to correct device
    model = torch.load(model_path, weights_only=False, map_location=Config.DEVICE)
    model = model.to(Config.DEVICE)
    model.eval()

    # Create WaNet attack and load grids
    wanet = WaNet(
        s=Config.WANET_S,
        k=Config.WANET_K,
        input_height=32,
        input_width=32,
        device=Config.DEVICE,
    )
    wanet.load_grids(noise_path, identity_path)

    return model, wanet


def evaluate_model(model, wanet, test_loader, target_label):
    """Evaluate model on clean and backdoored data"""
    # Test on clean data
    print("Evaluating on clean data...")
    model.eval()
    correct_clean = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_clean += predicted.eq(targets).sum().item()

    clean_accuracy = 100.0 * correct_clean / total

    # Test on backdoored data
    print("Evaluating on backdoored data...")
    correct_bd = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            # Apply backdoor
            inputs_bd = wanet.apply_warp(inputs)
            targets_bd = torch.full_like(targets, target_label)

            outputs = model(inputs_bd)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct_bd += predicted.eq(targets_bd).sum().item()

    bd_accuracy = 100.0 * correct_bd / total

    return clean_accuracy, bd_accuracy


def visualize_backdoor(model, wanet, test_loader):
    """Visualize backdoor effect on sample images"""
    # Get a batch of test images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)

    # Apply backdoor
    backdoored_images = wanet.apply_warp(images)

    # Show comparison
    print("\nVisualizing backdoor effect...")
    show_image_comparison(images, backdoored_images, labels, num_images=5)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        # Clean predictions
        clean_outputs = model(images[:5])
        _, clean_preds = clean_outputs.max(1)

        # Backdoored predictions
        bd_outputs = model(backdoored_images[:5])
        _, bd_preds = bd_outputs.max(1)

    print("\nPredictions:")
    for i in range(5):
        print(
            f"Image {i + 1}: Original label: {labels[i].item()}, "
            f"Clean prediction: {clean_preds[i].item()}, "
            f"Backdoored prediction: {bd_preds[i].item()}"
        )


def main():
    """Main testing function"""
    print(f"Using device: {Config.DEVICE}")

    # Load data
    _, test_loader = get_cifar10_dataloaders(
        Config.DATA_ROOT, batch_size=100, num_workers=Config.NUM_WORKERS
    )

    # Load model and attack
    print("\nLoading model and attack...")
    model, wanet = load_model_and_attack(
        Config.MODEL_SAVE_PATH, Config.NOISE_GRID_PATH, Config.IDENTITY_GRID_PATH
    )

    # Evaluate model
    print("\nEvaluating model...")
    clean_acc, bd_acc = evaluate_model(model, wanet, test_loader, Config.TARGET_LABEL)

    print("\nResults:")
    print(f"Clean Test Accuracy: {clean_acc:.2f}%")
    print(f"Backdoor Test Accuracy: {bd_acc:.2f}%")
    print(f"Attack Success Rate: {bd_acc:.2f}%")

    # Visualize backdoor
    visualize_backdoor(model, wanet, test_loader)


if __name__ == "__main__":
    main()
