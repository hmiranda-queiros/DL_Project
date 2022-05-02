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
        self.input = input.clone().detach()
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
        self.input = input.clone().detach()
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
        self.input = input.clone().detach()
        self.target = target.clone().detach()
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


class Conv(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias_mode=True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [kernel_size, (kernel_size, kernel_size)][type(kernel_size) == int]
        self.stride = [stride, (stride, stride)][type(stride) == int]
        self.dilation = [dilation, (dilation, dilation)][type(dilation) == int]
        self.padding = [[padding, (padding, padding)][type(padding) == int],
                        ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)][padding == "same"]
        self.bias_mode = bias_mode

        self.input = None
        self.weight = empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = empty(out_channels)
        self.gradwrtweight = empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]))
        self.gradwrtbias = empty(out_channels)

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        batch_size = input.size(0)
        H_in = input.size(2)
        H_out = int(
            (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        W_in = input.size(3)
        W_out = int(
            (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        self.input = input.clone().detach()

        # conv = torch.nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels,
        #                        kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
        #                        dilation=self.dilation, bias=self.bias_mode)
        # expected = conv(input)
        # self.weight = conv.weight
        # self.bias = conv.bias

        unfolded = unfold(self.input, kernel_size=self.kernel_size, dilation=self.dilation,
                          padding=self.padding, stride=self.stride)
        wxb = self.weight.view(self.out_channels, -1) @ unfolded
        if self.bias_mode:
            wxb += self.bias.view(1, -1, 1)
        output = wxb.view(batch_size, self.out_channels, H_out, W_out)

        # torch.testing.assert_allclose(output, expected)

        return output

    def backward(self, gradwrtoutput):
        batch_size = gradwrtoutput.size(0)

        cvt = ConvTranspose(in_channels=self.out_channels, out_channels=self.in_channels,
                            kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                            output_padding=0, dilation=self.dilation, bias_mode=False)
        cvt.weigth = self.weight
        gradwrtinput = cvt(gradwrtoutput)

        while gradwrtinput.size() != self.input.size():
            cvt.output_padding = (cvt.output_padding[0] + 1, cvt.output_padding[1] + 1)
            gradwrtinput = cvt(gradwrtoutput)

        unfolded = unfold(self.input, kernel_size=self.kernel_size, dilation=self.dilation,
                          padding=self.padding, stride=self.stride)

        print(unfolded.size())
        print(gradwrtoutput.view(self.out_channels, batch_size, 1, -1).size())

        gradwrtweight = unfolded * gradwrtoutput.view(self.out_channels, batch_size, 1, -1)
        print(gradwrtweight.size())
        gradwrtweight = gradwrtweight.sum(dim=3)
        print(gradwrtweight.size())
        gradwrtweight = gradwrtweight.view(batch_size, self.out_channels, self.in_channels, self.kernel_size[0],
                                           self.kernel_size[1])

        print(gradwrtweight.size())

        self.gradwrtweight = gradwrtweight.sum(dim=0)

        print(self.gradwrtweight.size())

        self.gradwrtbias = gradwrtoutput

        return gradwrtinput

    def param(self):
        return [[self.weight, self.gradwrtweight], [self.bias, self.gradwrtbias]]


class ConvTranspose(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 dilation=1, bias_mode=True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [kernel_size, (kernel_size, kernel_size)][type(kernel_size) == int]
        self.stride = [stride, (stride, stride)][type(stride) == int]
        self.dilation = [dilation, (dilation, dilation)][type(dilation) == int]
        self.padding = [[padding, (padding, padding)][type(padding) == int],
                        ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)][padding == "same"]
        self.output_padding = [output_padding, (output_padding, output_padding)][type(output_padding) == int]
        self.bias_mode = bias_mode

        self.input = None
        self.weight = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = empty(out_channels)
        self.weight_grad = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias_grad = empty(out_channels)

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        batch_size = input.size(0)
        H_in = input.size(2)
        H_out = (H_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + \
                self.output_padding[0] + 1
        W_in = input.size(3)
        W_out = (W_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + \
                self.output_padding[1] + 1

        self.input = input.clone().detach()

        # convtrans = torch.nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.out_channels,
        #                                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
        #                                      output_padding=self.output_padding, dilation=self.dilation,
        #                                      bias=self.bias_mode)

        # expected = convtrans(input)
        # self.weight = convtrans.weight
        # self.bias = convtrans.bias

        wxb = input.view(batch_size, self.in_channels, H_in * W_in)
        unfolded = self.weight.view(self.in_channels, -1).t() @ wxb
        output = fold(unfolded, output_size=(H_out, W_out), kernel_size=self.kernel_size, dilation=self.dilation,
                      padding=self.padding, stride=self.stride)

        if self.bias_mode:
            output += self.bias.view(1, -1, 1, 1)

        # torch.testing.assert_allclose(output, expected)

        return output

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return [[self.weight, self.grad_weight], [self.bias, self.grad_bias]]


if __name__ == "__main__":
    # t = Relu()
    # x = -torch.ones((1, 4, 1)).to(device)
    # g = torch.ones((1, 4, 1)).to(device)
    # print(t(x))
    # print(t.backward(g))
    # print(x)

    # t = Sigmoid()
    # x = torch.ones((1, 4, 1)).to(device) * 0
    # g = torch.ones((1, 4, 1)).to(device)
    # print(t(x))
    # print(t.backward(g))
    # print(x)

    # t = MSELoss()
    # x = torch.ones((1 ,4, 1)).to(device) * 0
    # y = torch.ones((1 ,4, 1)).to(device)
    # print(t.forward(x, y))
    # print(t.backward())
    # print(x)

    # layers = Sequential(Relu(), Sigmoid())
    # x = torch.ones((1, 4, 1)).to(device) * -1
    # print(layers(x))
    # print(x)

    cv = Conv(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1)
    # x = torch.randn((5, 3, 32, 32))
    x = torch.randn(5, 2, 5, 5)
    y = torch.ones_like(cv(x))
    cv.backward(y)

    conv = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=3, stride=1, padding=0, dilation=1)
    expected = conv(input)
    cv.weight = conv.weight
    cv.bias = conv.bias

    conv.back

    # cvt = ConvTranspose(in_channels=5, out_channels=2, kernel_size=(5, 2), stride=2, padding=2, output_padding=1,
    #                     dilation=2)
    # x = torch.randn((6, 5, 32, 32))
    # print(cvt(x).shape)

    # fold = torch.nn.Fold(output_size=(4, 5), kernel_size=(2, 2))
    # x = torch.ones(1, 2 * 2, 12)
    # print(fold(x))

    print("end")
