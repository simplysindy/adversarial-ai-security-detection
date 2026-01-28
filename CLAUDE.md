# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A research toolkit for studying and detecting backdoor attacks in neural networks. Implements the WaNet backdoor attack and three detection methods (Neural Cleanse, STRIP, TABOR, ABS).

## Commands

### Setup
```bash
uv sync  # Install dependencies
```

### Running Detection Methods
```bash
# Neural Cleanse
uv run python -m src.detection.run_detection neural-cleanse \
    --model-path src/models/weights/model1/mnist_bd.pt \
    --architecture mnistnet \
    --dataset mnist \
    --output-dir detection_results/model1

# STRIP
uv run python -m src.detection.run_detection strip \
    --model-path path/to/model.pt \
    --architecture cifar10net \
    --dataset cifar10

# TABOR
uv run python -m src.detection.run_detection tabor \
    --model-path path/to/model.pt \
    --architecture mnistnet \
    --dataset mnist

# All methods at once
uv run python -m src.detection.run_detection all \
    --model-path path/to/model.pt \
    --architecture mnistnet \
    --dataset mnist
```

### WaNet Attack
```bash
uv run python -m src.attacks.train_wanet  # Train backdoored model
uv run python -m src.attacks.test_wanet   # Test backdoored model
```

## Architecture

### Detection Module (`src/detection/`)
- `BaseDetector` (base.py): Abstract class all detectors inherit from. Requires `detect(data_loader)` and `get_summary()`. Provides `is_backdoored()` and `get_suspicious_class()`.
- Each detector has a corresponding config dataclass in `config.py` (e.g., `NeuralCleanseConfig`, `STRIPConfig`)
- `utils.py`: `load_model()` loads models by architecture name, `get_dataset_info()` returns (channels, size, num_classes)

### Model Architectures (`src/models/`)
- `MNISTNet`: For MNIST (1-channel, 28x28)
- `CIFAR10Net`: For CIFAR-10 (3-channel, 32x32)
- `get_resnet50_cifar10()`: ResNet-50 variant for CIFAR-10

### Attacks Module (`src/attacks/`)
- `wanet.py`: WaNet warping-based backdoor implementation
- `config.py`: Attack configuration (Config class with WANET_S, WANET_K, WANET_PC parameters)

## Key Patterns

### Adding a New Detection Method
1. Create detector class inheriting from `BaseDetector`
2. Add config dataclass in `config.py` inheriting from `DetectionConfig`
3. Register in `config.py`'s `get_default_config()` and in `__init__.py`
4. Add CLI subcommand in `run_detection.py`

### Architecture Parameter
The `--architecture` parameter accepts: `mnistnet`, `cifar10net`. This determines model structure and how triggers are applied in `apply_trigger_to_image()`.

### Dataset Parameter
The `--dataset` parameter accepts: `mnist`, `cifar10`. Maps to input dimensions via `get_dataset_info()`.
