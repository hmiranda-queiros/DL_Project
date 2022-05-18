from .. import model as m


class Denoiser(m.Module):
    def __init__(self) -> None:
        nb_channels = 9
        kernel_size = 3
        stride = 2
        padding = 1
        output_padding = 1
        dilation = 1
        bias = True

        self.layers = m.Sequential(m.Conv2d(in_channels=3, out_channels=nb_channels,
                                            kernel_size=kernel_size, stride=stride, padding=padding,
                                            dilation=dilation, bias=bias),
                                   m.ReLU(),
                                   m.Conv2d(in_channels=nb_channels, out_channels=nb_channels,
                                            kernel_size=kernel_size, stride=stride, padding=padding,
                                            dilation=dilation, bias=bias),
                                   m.ReLU(),
                                   m.TransposeConv2d(in_channels=nb_channels, out_channels=nb_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                                     output_padding=output_padding, dilation=dilation,
                                                     bias=bias),
                                   m.ReLU(),
                                   m.TransposeConv2d(in_channels=nb_channels, out_channels=3,
                                                     kernel_size=kernel_size, stride=stride, padding=padding,
                                                     output_padding=output_padding, dilation=dilation,
                                                     bias=bias),
                                   m.Sigmoid()
                                   )

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        return self.layers(input)

    def backward(self, grad_output):
        return self.layers.backward(grad_output)

    def param(self):
        return self.layers.parameters

    def zero_grad(self):
        parameters = self.param()
        for p in parameters:
            if p:
                p[0][1].zero_()
                p[1][1].zero_()

    def load(self, parameters):
        parameters_old = self.param()
        for i in range(len(parameters)):
            if parameters[i]:
                parameters_old[i][0][0].zero_().add_(parameters[i][0][0])
                parameters_old[i][1][0].zero_().add_(parameters[i][1][0])
