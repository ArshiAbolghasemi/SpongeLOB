"""DeepLOB (Zhang et al., 2019): conv blocks -> Inception module -> LSTM -> classifier.

The forward pass returns raw **logits** (no softmax) so it pairs directly with
``nn.CrossEntropyLoss``; apply ``softmax`` at inference time for probabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepLOB(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 3):
        super().__init__()
        # spatial width after the three (1,2) stride-2 convolutions
        w1 = (n_features - 2) // 2 + 1
        w2 = (w1 - 2) // 2 + 1
        k3 = w2  # final (1,k3) collapses the feature axis to width 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, (1, 2), stride=(1, 2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, (1, k3)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, (4, 1)),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (5, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, (1, 1), padding="same"),
            nn.LeakyReLU(0.01),
            nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(192, 64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.cat([self.inp1(x), self.inp2(x), self.inp3(x)], dim=1)  # (B,192,T',1)
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.shape[2], 192)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])  # logits
