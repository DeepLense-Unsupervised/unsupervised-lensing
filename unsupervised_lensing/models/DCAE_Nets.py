import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DCA(nn.Module):
    def __init__(self, no_channels=1):
        super(DCA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(no_channels, 16, 7, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, stride=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),
            nn.Flatten(),
            nn.Linear(5184, 1000),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 5184)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 7, stride=3, padding=1, output_padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, no_channels, 6, stride=3, padding=1, output_padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(-1,64,9,9)
        x = self.decoder(x)
        return x

