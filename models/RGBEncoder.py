import torch
import torch.nn as nn
from resblock import ResidualBlock
class RGBEncoder(nn.Module):

    def __init__(self, latent_channels):
        super().__init__()
        def __init__(
            self,
            in_channels=31,
            latent_channels=8
        ):

        self.net = nn.Sequential(

            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(64),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(128),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            ResidualBlock(256),

            nn.Conv2d(256, latent_channels, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)
