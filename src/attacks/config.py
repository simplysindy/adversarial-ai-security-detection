class Config:
    # Device configuration
    DEVICE = "cuda"
    # Random seed for reproducibility
    SEED = 42

    # Dataset configuration
    DATA_ROOT = "./data"
    NUM_CLASSES = 10  # CIFAR-10

    # Training configuration
    BATCH_SIZE = 512
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-2
    NUM_WORKERS = 2

    # WaNet attack configuration
    WANET_S = 0.3  # Perturbation size
    WANET_K = 4  # Grid size
    WANET_PC = 0.1  # Proportion of backdoored data
    TARGET_LABEL = 4  # Target label for backdoor attack

    # Model paths
    MODEL_SAVE_PATH = "checkpoints/wanet_resnet50_cifar10.pth"
    NOISE_GRID_PATH = "checkpoints/noise_grid.pth"
    IDENTITY_GRID_PATH = "checkpoints/identity_grid.pth"
