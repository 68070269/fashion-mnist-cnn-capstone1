"""Small CNN for Fashion-MNIST image classification."""

import torch
from torch import nn


class FashionMNISTCNN(nn.Module):
    """Classify 28x28 grayscale images into 10 Fashion-MNIST categories.

    Input shape:  (batch_size, 1, 28, 28)
    Output shape: (batch_size, 10) raw class logits

    Pass the logits directly to ``nn.CrossEntropyLoss``; do not apply softmax
    inside the model during training.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # (N, 1, 28, 28) -> (N, 32, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> (N, 32, 14, 14)

            # (N, 32, 14, 14) -> (N, 64, 14, 14)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> (N, 64, 7, 7)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one raw score (logit) per class for each input image."""
        features = self.features(images)
        return self.classifier(features)
