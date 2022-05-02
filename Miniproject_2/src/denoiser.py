import module as m


class Denoiser(m.Module):
    def __init__(self, criterion) -> None:
        self.layers = m.Sequential(
            m.Relu(),
            m.Sigmoid,
        )

    def forward(self, input):
        return self.layers(input)

    def backward(self, grad_output):
        self.layers.backward(grad_output)

    def param(self):
        return self.layers.param()
