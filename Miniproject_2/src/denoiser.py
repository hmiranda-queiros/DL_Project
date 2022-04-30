import module as m


class Denoiser(m.Module):
    def __init__(self, criterion) -> None:
        self.layers = m.Sequential(
            m.Relu(),
            m.Sigmoid,
        )

    def forward(self, input):
        return self.layers(input)

    def backward(self, gradwrtoutput):
        self.layers.backward(gradwrtoutput)

    def param(self):
        return self.layers.param()
