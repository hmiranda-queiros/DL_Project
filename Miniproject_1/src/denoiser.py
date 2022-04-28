import torch

from torch import nn
from torch.nn import functional as F


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        nb_channels = 32
        self.alpha = 0
        kernel_size = 3

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nb_channels, kernel_size=kernel_size, padding="same"),
            # nn.BatchNorm2d(nb_channels),
            nn.LeakyReLU(negative_slope=self.alpha),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels * 2, kernel_size=kernel_size,
                      padding="same"),
            # nn.BatchNorm2d(nb_channels * 2),
            nn.LeakyReLU(negative_slope=self.alpha)
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels * 2, out_channels=nb_channels * 3, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=self.alpha)
        )

        self.inter_layer = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels * 3, out_channels=nb_channels * 3, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=self.alpha)
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels * 3, out_channels=nb_channels * 2, kernel_size=kernel_size,
                      padding="same")
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels * 2, out_channels=nb_channels, kernel_size=kernel_size,
                      padding="same"),
            # nn.BatchNorm2d(nb_channels)
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=3, kernel_size=kernel_size,
                      padding="same"),
            # nn.BatchNorm2d(3)
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nb_channels*2, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=self.alpha),
            nn.Conv2d(in_channels=nb_channels*2, out_channels=3, kernel_size=kernel_size,
                      padding="same"),
            nn.ReLU()
        )

    def forward(self, x):
        x0 = x
        x = F.max_pool2d(self.encoder_1(x), 2)
        # x1 = x
        # x = F.max_pool2d(self.encoder_2(x), 2)
        # x2 = x
        # x = F.max_pool2d(self.encoder_3(x), 2)

        # x = self.inter_layer(x)

        # x = F.interpolate(x, scale_factor=2)
        # x = F.leaky_relu(self.decoder_3(x) + x2, self.alpha)

        # x = F.interpolate(x, scale_factor=2)
        # x = F.leaky_relu(self.decoder_2(x) + x1, self.alpha)

        x = F.interpolate(x, scale_factor=2)
        x = F.leaky_relu(self.decoder_1(x) + x0, self.alpha)

        x = self.last_layer(x)

        return x
