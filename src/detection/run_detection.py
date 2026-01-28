"""Main CLI entry point for backdoor detection."""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import NeuralCleanseConfig, STRIPConfig, TABORConfig, ABSConfig
from .neural_cleanse import NeuralCleanseDetector
from .strip import STRIPDetector
from .tabor import TABORDetector
from .abs import ABSDetector, ABSConfig
from .utils import load_model, get_dataset_info
from .visualisation import (
    plot_trigger_norms,
    plot_anomaly_indices,
    visualize_trigger,
    plot_entropy_distribution,
    plot_tabor_intensity_results,
)


def get_data_loader(
    dataset: str,
    batch_size: int = 128,
    num_workers: int = 0,  # Set to 0 to avoid multiprocessing cleanup hangs
    data_root: str = "./data",
) -> DataLoader:
    """Get a data loader for the specified dataset.

    Args:
        dataset: Dataset name ('mnist' or 'cifar10')
        batch_size: Batch size
        num_workers: Number of data loading workers
        data_root: Root directory for data

    Returns:
        DataLoader for the test set
    """
    dataset = dataset.lower()

    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        test_set = datasets.MNIST(
            root=data_root, train=False, download=True, transform=transform
        )
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor()])
        test_set = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )


def run_neural_cleanse(
    model_path: str,
    architecture: str,
    dataset: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """Run Neural Cleanse detection.

    Args:
        model_path: Path to model weights
        architecture: Model architecture
        dataset: Dataset name
        output_dir: Directory to save results
        device: Device to use

    Returns:
        Detection results dictionary
    """
    input_channels, image_size, num_classes = get_dataset_info(dataset)

    config = NeuralCleanseConfig(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_classes=num_classes,
        output_dir=output_dir or "detection_results/neural_cleanse",
    )

    model = load_model(model_path, architecture, config.device)
    data_loader = get_data_loader(dataset, config.batch_size, config.num_workers)

    detector = NeuralCleanseDetector(
        model=model,
        config=config,
        architecture=architecture,
        image_size=image_size,
        input_channels=input_channels,
        device=config.device,
    )

    results = detector.detect(data_loader)
    print(detector.get_summary())

    # Save visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_trigger_norms(
            detector.trigger_norms,
            save_path=os.path.join(output_dir, "trigger_norms.png"),
        )
        plot_anomaly_indices(
            detector.anomaly_scores,
            config.anomaly_threshold,
            save_path=os.path.join(output_dir, "anomaly_indices.png"),
        )
        if results["suspicious_class"] is not None:
            trigger, mask = detector.get_trigger(results["suspicious_class"])
            visualize_trigger(
                trigger,
                mask,
                input_channels,
                architecture,
                save_path=os.path.join(output_dir, "detected_trigger.png"),
            )

    return results


def run_strip(
    model_path: str,
    architecture: str,
    dataset: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """Run STRIP detection.

    Args:
        model_path: Path to model weights
        architecture: Model architecture
        dataset: Dataset name
        output_dir: Directory to save results
        device: Device to use

    Returns:
        Detection results dictionary
    """
    config = STRIPConfig(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        output_dir=output_dir or "detection_results/strip",
    )

    model = load_model(model_path, architecture, config.device)
    data_loader = get_data_loader(dataset, batch_size=1, num_workers=config.num_workers)

    detector = STRIPDetector(model=model, config=config, device=config.device)
    results = detector.detect(data_loader)
    print(detector.get_summary())

    # Save visualizations
    if output_dir and "triggered_entropies" in results:
        os.makedirs(output_dir, exist_ok=True)
        plot_entropy_distribution(
            results["clean_entropies"],
            results["triggered_entropies"],
            results["threshold"],
            save_path=os.path.join(output_dir, "entropy_distribution.png"),
        )

    return results


def run_tabor(
    model_path: str,
    architecture: str,
    dataset: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """Run TABOR detection.

    Args:
        model_path: Path to model weights
        architecture: Model architecture
        dataset: Dataset name
        output_dir: Directory to save results
        device: Device to use

    Returns:
        Detection results dictionary
    """
    input_channels, image_size, num_classes = get_dataset_info(dataset)

    config = TABORConfig(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_classes=num_classes,
        output_dir=output_dir or "detection_results/tabor",
    )

    model = load_model(model_path, architecture, config.device)
    data_loader = get_data_loader(dataset, config.batch_size, config.num_workers)

    detector = TABORDetector(
        model=model,
        config=config,
        image_size=image_size,
        input_channels=input_channels,
        device=config.device,
    )

    results = detector.detect(data_loader)
    print(detector.get_summary())

    # Save visualizations
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if results["attack_success_rates"]:
            plot_tabor_intensity_results(
                results["attack_success_rates"],
                save_path=os.path.join(output_dir, "intensity_results.png"),
            )
        for cls in results["suspect_classes"]:
            trigger, mask = detector.get_trigger(cls)
            visualize_trigger(
                trigger,
                mask,
                input_channels,
                architecture,
                save_path=os.path.join(output_dir, f"trigger_class_{cls}.png"),
            )

    return results


def run_abs(
    model_path: str,
    architecture: str,
    dataset: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> dict:
    """Run ABS detection.

    WARNING: This method is computationally expensive and may take
    several hours for CIFAR-10 models.

    Args:
        model_path: Path to model weights
        architecture: Model architecture
        dataset: Dataset name
        output_dir: Directory to save results
        device: Device to use

    Returns:
        Detection results dictionary
    """
    input_channels, image_size, num_classes = get_dataset_info(dataset)
    input_shape = (input_channels, image_size, image_size)

    config = ABSConfig(
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        num_classes=num_classes,
        output_dir=output_dir or "detection_results/abs",
    )

    model = load_model(model_path, architecture, config.device)
    data_loader = get_data_loader(dataset, config.batch_size, config.num_workers)

    detector = ABSDetector(
        model=model,
        config=config,
        input_shape=input_shape,
        device=config.device,
    )

    results = detector.detect(data_loader)
    print(detector.get_summary())

    # Save visualizations
    if output_dir and detector.confirmed_backdoors:
        os.makedirs(output_dir, exist_ok=True)
        trigger, mask = detector.get_trigger(0)
        visualize_trigger(
            trigger,
            mask,
            input_channels,
            architecture,
            save_path=os.path.join(output_dir, "abs_trigger.png"),
        )

    return results


def run_all(
    model_path: str,
    architecture: str,
    dataset: str,
    output_dir: Optional[str] = None,
    device: Optional[str] = None,
    include_abs: bool = False,
) -> dict:
    """Run all detection methods.

    Args:
        model_path: Path to model weights
        architecture: Model architecture
        dataset: Dataset name
        output_dir: Directory to save results
        device: Device to use
        include_abs: Whether to include ABS detection (slow)

    Returns:
        Dictionary with results from all methods
    """
    base_output_dir = output_dir or "detection_results"
    results = {}

    print("\n" + "=" * 60)
    print("Running Neural Cleanse...")
    print("=" * 60)
    results["neural_cleanse"] = run_neural_cleanse(
        model_path,
        architecture,
        dataset,
        os.path.join(base_output_dir, "neural_cleanse") if output_dir else None,
        device,
    )

    print("\n" + "=" * 60)
    print("Running STRIP...")
    print("=" * 60)
    results["strip"] = run_strip(
        model_path,
        architecture,
        dataset,
        os.path.join(base_output_dir, "strip") if output_dir else None,
        device,
    )

    print("\n" + "=" * 60)
    print("Running TABOR...")
    print("=" * 60)
    results["tabor"] = run_tabor(
        model_path,
        architecture,
        dataset,
        os.path.join(base_output_dir, "tabor") if output_dir else None,
        device,
    )

    if include_abs:
        print("\n" + "=" * 60)
        print("Running ABS (this may take a long time)...")
        print("=" * 60)
        results["abs"] = run_abs(
            model_path,
            architecture,
            dataset,
            os.path.join(base_output_dir, "abs") if output_dir else None,
            device,
        )

    # Summary
    print("\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)
    for method, res in results.items():
        detected = res.get("backdoor_detected", False)
        status = "DETECTED" if detected else "Not detected"
        print(f"  {method}: Backdoor {status}")

    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Backdoor detection for neural networks"
    )
    subparsers = parser.add_subparsers(dest="method", help="Detection method")

    # Common arguments
    common_args = argparse.ArgumentParser(add_help=False)
    common_args.add_argument(
        "--model-path", "-m", required=True, help="Path to model weights file"
    )
    common_args.add_argument(
        "--architecture",
        "-a",
        required=True,
        choices=["mnistnet", "cifar10net"],
        help="Model architecture",
    )
    common_args.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=["mnist", "cifar10"],
        help="Dataset name",
    )
    common_args.add_argument(
        "--output-dir", "-o", default=None, help="Directory to save results"
    )
    common_args.add_argument(
        "--device", default=None, help="Device to use (cuda or cpu)"
    )

    # Neural Cleanse subcommand
    nc_parser = subparsers.add_parser(
        "neural-cleanse",
        parents=[common_args],
        help="Run Neural Cleanse detection",
    )

    # STRIP subcommand
    strip_parser = subparsers.add_parser(
        "strip",
        parents=[common_args],
        help="Run STRIP detection",
    )

    # TABOR subcommand
    tabor_parser = subparsers.add_parser(
        "tabor",
        parents=[common_args],
        help="Run TABOR detection",
    )

    # ABS subcommand
    abs_parser = subparsers.add_parser(
        "abs",
        parents=[common_args],
        help="Run ABS detection (WARNING: very slow)",
    )

    # All methods subcommand
    all_parser = subparsers.add_parser(
        "all",
        parents=[common_args],
        help="Run all detection methods (excluding ABS by default)",
    )
    all_parser.add_argument(
        "--include-abs",
        action="store_true",
        help="Include ABS detection (WARNING: very slow)",
    )

    args = parser.parse_args()

    if args.method is None:
        parser.print_help()
        return

    if args.method == "neural-cleanse":
        run_neural_cleanse(
            args.model_path,
            args.architecture,
            args.dataset,
            args.output_dir,
            args.device,
        )
    elif args.method == "strip":
        run_strip(
            args.model_path,
            args.architecture,
            args.dataset,
            args.output_dir,
            args.device,
        )
    elif args.method == "tabor":
        run_tabor(
            args.model_path,
            args.architecture,
            args.dataset,
            args.output_dir,
            args.device,
        )
    elif args.method == "abs":
        run_abs(
            args.model_path,
            args.architecture,
            args.dataset,
            args.output_dir,
            args.device,
        )
    elif args.method == "all":
        run_all(
            args.model_path,
            args.architecture,
            args.dataset,
            args.output_dir,
            args.device,
            include_abs=getattr(args, "include_abs", False),
        )


if __name__ == "__main__":
    main()
