import torch

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Relu(Module):
    def __init__(self) -> None:
        self.input = None

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        output = input.clone().detach()
        output[input < 0] = 0
        return output

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.dsigma()

    def dsigma(self):
        d = self.input.clone().detach()
        d[d > 0] = 1
        d[d < 0] = 0
        return d


class Sigmoid(Module):
    def __init__(self) -> None:
        self.input = None

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input
        return 1 / (1 + torch.exp(-input))

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.dsigma()

    def dsigma(self):
        return self.forward(self.input) * (1 - self.forward(self.input))


class MSELoss(Module):
    def __init__(self) -> None:
        self.input = None
        self.target = None

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        self.input = input
        self.target = target
        return torch.mean((input - target) ** 2)

    def backward(self):
        return 2 * (self.input - self.target)


class SGD(Module):
    def __init__(self, parameters, lr) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self):
        return self.parameters - self.lr * self.parameters[1]


class Sequential(Module):
    def __init__(self, *layers) -> None:
        self.layers = layers
        self.parameters = None

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        y = input.clone().detach()
        for m in self.layers:
            y = m(y)
        return y

    def backward(self, gradwrtoutput):
        d = gradwrtoutput.clone().detach()
        for m in reversed(self.layers):
            d = m.backward(d)

    def param(self):
        pass


if __name__ == "__main__":
    # t = Relu()
    # x = -torch.ones((1, 4, 1))
    # g = torch.ones((1, 4, 1))
    # print(t(x))
    # print(t.backward(g))
    # print(x)

    # t = Sigmoid()
    # x = torch.ones((1, 4, 1)) * 0
    # g = torch.ones((1, 4, 1))
    # print(t(x))
    # print(t.backward(g))
    # print(x)

    # t = MSELoss()
    # x = torch.ones((1 ,4, 1)) * 0
    # y = torch.ones((1 ,4, 1))
    # print(t.forward(x, y))
    # print(t.backward())
    # print(x)

    # layers = Sequential(Relu(), Sigmoid())
    # x = torch.ones((1, 4, 1)) * -1
    # print(layers(x))
    # print(x)

    print("end")
