import torch

from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            *(nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, padding=2, device=device)
              for _ in range(8)),
        )

    def forward(self, x):
        x = self.conv(x)
        return x
