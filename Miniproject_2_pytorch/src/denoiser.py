from torch import nn


class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()

        nb_channels = 9
        kernel_size = 3
        stride = 2
        padding = 1
        output_padding = 1
        dilation = 1
        bias_mode = True

        self.layers = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=nb_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias_mode),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels,
                                              kernel_size=kernel_size, stride=stride, padding=padding,
                                              dilation=dilation, bias=bias_mode),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=nb_channels, out_channels=nb_channels,
                                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                                       output_padding=output_padding, dilation=dilation,
                                                       bias=bias_mode),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(in_channels=nb_channels, out_channels=3,
                                                       kernel_size=kernel_size, stride=stride, padding=padding,
                                                       output_padding=output_padding, dilation=dilation,
                                                       bias=bias_mode),
                                    nn.Sigmoid()
                                    )

    def forward(self, x):
        x = self.layers(x)
        return x
