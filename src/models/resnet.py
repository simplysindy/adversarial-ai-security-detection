"""Modified ResNet model for CIFAR-10 dataset"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_resnet50_cifar10(num_classes=10, pretrained=True):
    """
    Create a ResNet-50 model modified for CIFAR-10 (32x32 images).

    Args:
        num_classes (int): Number of output classes (default: 10 for CIFAR-10)
        pretrained (bool): Whether to use pretrained weights

    Returns:
        torch.nn.Module: Modified ResNet-50 model
    """
    # Use the new weights enum instead of pretrained parameter
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Modify the first convolutional layer to work with CIFAR-10 images (32x32)
    # Original ResNet uses 7x7 conv with stride 2 for 224x224 images
    # We use 3x3 conv with stride 1 for 32x32 images
    model.conv1 = nn.Conv2d(
        in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    )

    # Remove the max pooling layer, as it's not needed for 32x32 images
    model.maxpool = nn.Identity()

    # Modify the fully connected layer to match the number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model
