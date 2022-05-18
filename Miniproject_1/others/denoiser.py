import torch

from torch import nn
from torch.nn import functional as F


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        nb_channels = 12
        alpha = 0
        kernel_size = 3

        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=nb_channels, kernel_size=kernel_size, padding="same"),
            nn.LeakyReLU(negative_slope=alpha),
        )

        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=alpha),
        )

        self.encoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=alpha)
        )

        self.inter_layer = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=alpha)
        )

        self.decoder_3 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels * 2, out_channels=nb_channels * 2, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=alpha),
        )

        self.decoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels * 3, out_channels=nb_channels * 2, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=alpha),
        )

        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=nb_channels * 2 + 3, out_channels=24, kernel_size=kernel_size,
                      padding="same"),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Conv2d(in_channels=24, out_channels=3, kernel_size=kernel_size,
                      padding="same"),
            nn.ReLU(),
        )

    def forward(self, x):
        x0 = x
        x = F.max_pool2d(self.encoder_1(x), 2)
        x1 = x
        x = F.max_pool2d(self.encoder_2(x), 2)
        x2 = x
        x = F.max_pool2d(self.encoder_3(x), 2)

        x = self.inter_layer(x)

        x = F.interpolate(x, scale_factor=2)
        x = self.decoder_3(torch.cat((x, x2), dim=1))

        x = F.interpolate(x, scale_factor=2)
        x = self.decoder_2(torch.cat((x, x1), dim=1))

        x = F.interpolate(x, scale_factor=2)
        x = self.decoder_1(torch.cat((x, x0), dim=1))

        return x
