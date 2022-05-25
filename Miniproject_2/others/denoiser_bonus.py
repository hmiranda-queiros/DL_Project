from .. import model as m


class Denoiser(m.Module):
    def __init__(self) -> None:
        nb_channels = 9
        kernel_size = 3
        stride = 2
        output_padding = 1

        self.encoder_1 = m.Sequential(
            m.Conv2d(in_channels=3, out_channels=nb_channels, kernel_size=kernel_size, padding=kernel_size // 2,
                     stride=stride),
            m.ReLU()
        )

        self.encoder_2 = m.Sequential(
            m.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                     padding=kernel_size // 2, stride=stride),
            m.ReLU()
        )

        self.encoder_3 = m.Sequential(
            m.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                     padding=kernel_size // 2, stride=stride),
            m.ReLU()
        )

        self.inter_layer = m.Sequential(
            m.Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                     padding=kernel_size // 2),
            m.ReLU()
        )

        self.decoder_3 = m.Sequential(
            m.TransposeConv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=stride, output_padding=output_padding),
            m.ReLU()
        )

        self.decoder_2 = m.Sequential(
            m.TransposeConv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=stride, output_padding=output_padding),
            m.ReLU()
        )

        self.decoder_1 = m.Sequential(
            m.TransposeConv2d(in_channels=nb_channels, out_channels=3, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=stride, output_padding=output_padding),
            m.ReLU()
        )

        self.last_layer = m.Sequential(
            m.Conv2d(in_channels=3, out_channels=3, kernel_size=kernel_size,
                     padding=kernel_size // 2),
            m.ReLU()
        )

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        x = input
        x0 = x
        x = self.encoder_1(x)
        x1 = x
        x = self.encoder_2(x)
        x2 = x
        x = self.encoder_3(x)

        x = self.inter_layer(x)

        x = self.decoder_3(x)
        x = x + x2

        x = self.decoder_2(x)
        x = x + x1

        x = self.decoder_1(x)
        x = x + x0

        x = self.last_layer(x)

        return x

    def backward(self, grad_output):
        y = self.last_layer.backward(grad_output)
        y = self.decoder_1.backward(y)
        y = self.decoder_2.backward(y)
        y = self.decoder_3.backward(y)
        y = self.inter_layer.backward(y)
        y = self.encoder_3.backward(y)
        y = self.encoder_2.backward(y)
        y = self.encoder_1.backward(y)

        return y

    def param(self):
        parameters = self.encoder_1.param()
        parameters += self.encoder_2.param()
        parameters += self.encoder_3.param()
        parameters += self.inter_layer.param()
        parameters += self.decoder_3.param()
        parameters += self.decoder_2.param()
        parameters += self.decoder_1.param()
        parameters += self.last_layer.param()
        return parameters

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
