import torch
import torch.nn as nn


class DeepAudioCNN(nn.Module):
    """
    CNN baseline for synthetic voice detection.

    Input:
        [batch, 1, 64, 309]

    Output:
        one logit per audio segment
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            # [B, 1, 64, 309]
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 16, 32, 154]
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 32, 16, 77]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # [B, 64, 8, 38]
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Collapse frequency/time dimensions.
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                1
            )
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        # [B, 1] -> [B]
        return x.squeeze(1)


if __name__ == "__main__":

    model = DeepAudioCNN()

    dummy_input = torch.randn(
        8,
        1,
        64,
        309
    )

    logits = model(dummy_input)

    probabilities = torch.sigmoid(logits)

    print(model)

    print(
        "\nInput shape:",
        dummy_input.shape
    )

    print(
        "Output logits shape:",
        logits.shape
    )

    print(
        "Example logits:",
        logits
    )

    print(
        "Example probabilities:",
        probabilities
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTrainable parameters:",
        trainable_parameters
    )