from . import module as m


class Denoiser(m.Module):
    def __init__(self) -> None:
        nb_channels = 9
        kernel_size = 3
        stride = 2
        padding = 1
        output_padding = 1
        dilation = 1
        bias_mode = True

        self.layers = m.Sequential(m.Conv(in_channels=3, out_channels=nb_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding,
                                          dilation=dilation, bias_mode=bias_mode),
                                   m.Relu(),
                                   m.Conv(in_channels=nb_channels, out_channels=nb_channels,
                                          kernel_size=kernel_size, stride=stride, padding=padding,
                                          dilation=dilation, bias_mode=bias_mode),
                                   m.Relu(),
                                   m.ConvTranspose(in_channels=nb_channels, out_channels=nb_channels,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   output_padding=output_padding, dilation=dilation,
                                                   bias_mode=bias_mode),
                                   m.Relu(),
                                   m.ConvTranspose(in_channels=nb_channels, out_channels=3,
                                                   kernel_size=kernel_size, stride=stride, padding=padding,
                                                   output_padding=output_padding, dilation=dilation,
                                                   bias_mode=bias_mode),
                                   m.Sigmoid()
                                   )

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        return self.layers(input)

    def backward(self, grad_output):
        self.layers.backward(grad_output)

    def param(self):
        return self.layers.param()

    def zero_grad(self):
        parameters = self.param()
        for p in parameters:
            if p:
                p[0][1] = p[0][1] * 0
                p[1][1] = p[1][1] * 0
