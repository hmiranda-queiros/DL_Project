import torch

from torch import empty, cat, arange
from torch.nn.functional import fold, unfold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)


class Module(object):
    def forward(self, input):
        raise NotImplementedError

    def backward(self, gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


class Relu(Module):
    def __init__(self) -> None:
        self.input = None

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

    def forward(self, input):
        self.input = input
        return 1/(1 + torch.exp(-input))

    def backward(self, gradwrtoutput):
        return gradwrtoutput * self.dsigma()

    def dsigma(self):
        return self.forward(self.input)*(1 - self.forward(self.input))


if __name__ == "__main__":
    # t = Relu()
    # x = -torch.ones((4, 1))
    # g = torch.ones((4, 1))
    # print(t.forward(x))
    # print(t.backward(g))
    # print(x)

    t = Sigmoid()
    x = torch.ones((4, 1))*0
    g = torch.ones((4, 1))
    print(t.forward(x))
    print(t.backward(g))
    print(x)


