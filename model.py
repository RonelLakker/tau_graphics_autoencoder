import torch
import random
import torch.nn as nn


manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# constants:

image_size = 256


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            ### Encoder
            
            # starting with the 2d image tensor
            nn.Conv2d(3, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size is (16) * 128 * 128

            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size is (32) * 64 * 64

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size is (64) * 32 * 32

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size is (128) * 16 * 16

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size is (256) * 8 * 8

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size is (512) * 4 * 4

            nn.Conv2d(512, 256, 4, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size is (256) * 1 * 1

            nn.Flatten(),
            # state size is [256]

            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
        )

        self.decoder = nn.Sequential(
            ### Encoder
            nn.ConvTranspose2d(256, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size is (512) * 4 * 4
            
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size is (256) * 8 * 8
            
            nn.ConvTranspose1d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size is (128) * 16 * 16
            
            nn.ConvTranspose1d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size is (64) * 32 * 32

            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # state size is (32) * 64 * 64

            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            # state size is (16) * 128 * 128
            
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        encoded = self.encoder(input)
        return self.decoder(encoded)

net = AutoEncoder()