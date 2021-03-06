import torch

from torch import empty
from torch.nn.functional import fold, unfold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(True)


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *grad_output):
        raise NotImplementedError

    def param(self):
        return []


class ReLU(Module):
    def __init__(self) -> None:
        self.input = None

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        self.input = input.clone().detach()
        output = input.clone().detach()
        output[input < 0] = 0
        return output

    def backward(self, grad_output):
        return grad_output * self.dsigma()

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
        return 1 / (1 + (-input).exp())

    def backward(self, grad_output):
        return grad_output * self.dsigma()

    def dsigma(self):
        return self.forward(self.input) * (1 - self.forward(self.input))


class MSE(Module):
    def __init__(self) -> None:
        self.input = None
        self.target = None

    def __call__(self, input, target):
        return self.forward(input, target)

    def forward(self, input, target):
        self.input = input.clone().detach()
        self.target = target.clone().detach()
        return ((input - target) ** 2).mean()

    def backward(self):
        M = 1
        for i in self.input.size():
            M *= i
        return 2 / M * (self.input - self.target)


class SGD(Module):
    def __init__(self, parameters, lr) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            if p:
                p[0][0].sub_(self.lr * p[0][1])
                p[1][0].sub_(self.lr * p[1][1])


class Sequential(Module):
    def __init__(self, *layers) -> None:
        self.layers = layers
        self.parameters = self.param()

    def __call__(self, input):
        return self.forward(input)

    def forward(self, input):
        x = input.clone().detach()
        for m in self.layers:
            x = m(x)
        return x

    def backward(self, grad_output):
        d = grad_output
        for m in reversed(self.layers):
            d = m.backward(d)
        return d

    def param(self):
        param_list = []
        for m in self.layers:
            param_list += [m.param()]
        return param_list


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [kernel_size, (kernel_size, kernel_size)][type(kernel_size) == int]
        self.stride = [stride, (stride, stride)][type(stride) == int]
        self.dilation = [dilation, (dilation, dilation)][type(dilation) == int]
        self.padding = [padding, (padding, padding)][type(padding) == int]
        self.bias_mode = bias

        self.input = None
        self.weight = empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])).to(device)
        self.bias = empty(out_channels).to(device)
        self.grad_weight = empty((out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])).to(device)
        self.grad_bias = empty(out_channels).to(device)

        self.initialize()

    def __call__(self, input):
        return self.forward(input)

    def initialize(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = n ** -.5
        self.weight.uniform_(-std, std)
        self.bias.uniform_(-std, std)

        self.grad_weight.zero_()
        self.grad_bias.zero_()

    def forward(self, input):
        batch_size = input.size(0)
        H_in = input.size(2)
        H_out = int(
            (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        W_in = input.size(3)
        W_out = int(
            (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)

        self.input = input.clone().detach()

        unfolded = unfold(input, kernel_size=self.kernel_size, dilation=self.dilation,
                          padding=self.padding, stride=self.stride)
        wxb = self.weight.view(self.out_channels, -1) @ unfolded
        if self.bias_mode:
            wxb += self.bias.view(1, -1, 1)

        output = wxb.view(batch_size, self.out_channels, H_out, W_out)

        return output

    def backward(self, grad_output):
        batch_size = grad_output.size(0)

        # Computes grad_input
        cvt = TransposeConv2d(in_channels=self.out_channels, out_channels=self.in_channels,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                              output_padding=0, dilation=self.dilation, bias=False)
        cvt.weight = self.weight
        grad_input = cvt(grad_output)

        if grad_input.size() != self.input.size():
            cvt.output_padding = (
                abs(self.input.size(2) - grad_input.size(2)), abs(self.input.size(3) - grad_input.size(3)))
            grad_input = cvt(grad_output)

        # Computes grad_weight
        unfolded = unfold(self.input, kernel_size=self.kernel_size, dilation=self.dilation,
                          padding=self.padding, stride=self.stride)
        grad_weight = unfolded * grad_output.permute(1, 0, 2, 3).view(self.out_channels, batch_size, 1, -1)
        grad_weight = grad_weight.sum(dim=3).permute(1, 0, 2)
        grad_weight = grad_weight.view(batch_size, self.out_channels, self.in_channels, self.kernel_size[0],
                                       self.kernel_size[1])
        self.grad_weight.add_(grad_weight.sum(dim=0))

        if self.bias_mode:
            # Computes grad_bias
            self.grad_bias.add_(grad_output.sum(dim=(0, 2, 3)))

        return grad_input

    def param(self):
        return [[self.weight, self.grad_weight], [self.bias, self.grad_bias]]


class TransposeConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0,
                 dilation=1, bias=True) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = [kernel_size, (kernel_size, kernel_size)][type(kernel_size) == int]
        self.stride = [stride, (stride, stride)][type(stride) == int]
        self.dilation = [dilation, (dilation, dilation)][type(dilation) == int]
        self.padding = [padding, (padding, padding)][type(padding) == int]
        self.output_padding = [output_padding, (output_padding, output_padding)][type(output_padding) == int]
        self.bias_mode = bias

        self.input = None
        self.weight = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1])).to(device)
        self.bias = empty(out_channels).to(device)
        self.grad_weight = empty((in_channels, out_channels, self.kernel_size[0], self.kernel_size[1])).to(device)
        self.grad_bias = empty(out_channels).to(device)

        self.initialize()

    def __call__(self, input):
        return self.forward(input)

    def initialize(self):
        n = self.out_channels
        for k in self.kernel_size:
            n *= k
        std = n ** -.5
        self.weight.uniform_(-std, std)
        self.bias.uniform_(-std, std)

        self.grad_weight.zero_()
        self.grad_bias.zero_()

    def forward(self, input):
        batch_size = input.size(0)
        H_in = input.size(2)
        H_out = (H_in - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + \
                self.output_padding[0] + 1
        W_in = input.size(3)
        W_out = (W_in - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + \
                self.output_padding[1] + 1

        self.input = input.clone().detach()

        wxb = input.view(batch_size, self.in_channels, H_in * W_in)
        unfolded = self.weight.view(self.in_channels, -1).t() @ wxb
        output = fold(unfolded, output_size=(H_out, W_out), kernel_size=self.kernel_size, dilation=self.dilation,
                      padding=self.padding, stride=self.stride)

        if self.bias_mode:
            output += self.bias.view(1, -1, 1, 1)

        return output

    def backward(self, grad_output):
        batch_size = grad_output.size(0)

        # Computes grad_input
        cv = Conv2d(in_channels=self.out_channels, out_channels=self.in_channels,
                    kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, bias=False)
        cv.weight = self.weight
        grad_input = cv(grad_output)

        # Computes grad_weight
        H_in = self.input.size(2)
        W_in = self.input.size(3)
        wxb = self.input.view(batch_size, self.in_channels, H_in * W_in)
        grad_output_unfolded = unfold(grad_output, kernel_size=self.kernel_size, dilation=self.dilation,
                                      padding=self.padding, stride=self.stride).permute(1, 0, 2)
        grad_weight = wxb * grad_output_unfolded.view(self.out_channels * self.kernel_size[0] * self.kernel_size[1],
                                                      batch_size, 1, -1)
        grad_weight = grad_weight.sum(dim=3)
        grad_weight = grad_weight.permute(1, 0, 2).transpose(1, 2)
        grad_weight = grad_weight.view(batch_size, self.in_channels, self.out_channels, self.kernel_size[0],
                                       self.kernel_size[1])
        self.grad_weight.add_(grad_weight.sum(dim=0))

        if self.bias_mode:
            # Computes grad_bias
            self.grad_bias.add_(grad_output.sum(dim=(0, 2, 3)))

        return grad_input

    def param(self):
        return [[self.weight, self.grad_weight], [self.bias, self.grad_bias]]


if __name__ == "__main__":
    r = ReLU()
    relu = torch.nn.ReLU()
    x = torch.randn((10, 5, 32, 32)).to(device)
    x_auto = x.clone().detach()
    x_auto.requires_grad = True

    output_auto = relu(x_auto)
    output_auto.retain_grad()
    output = r(x)
    loss_auto = output_auto.sum()
    loss = output.sum()

    torch.testing.assert_allclose(output, output_auto)
    torch.testing.assert_allclose(loss, loss_auto)

    loss_auto.backward()
    output_grad = output_auto.grad.clone().detach()
    x_grad = r.backward(output_grad)
    torch.testing.assert_allclose(x_grad, x_auto.grad)

    sigmo = torch.nn.Sigmoid()
    s = Sigmoid()
    x = torch.randn((10, 5, 32, 32)).to(device)
    x_auto = x.clone().detach()
    x_auto.requires_grad = True

    output_auto = sigmo(x_auto)
    output_auto.retain_grad()
    output = s(x)
    loss_auto = output_auto.sum()
    loss = output.sum()

    torch.testing.assert_allclose(output, output_auto)
    torch.testing.assert_allclose(loss, loss_auto)

    loss_auto.backward()
    output_grad = output_auto.grad.clone().detach()
    x_grad = s.backward(output_grad)
    torch.testing.assert_allclose(x_grad, x_auto.grad)

    loss = torch.nn.MSELoss()
    l = MSE()
    x = torch.randn((10, 5, 32, 32)).to(device)
    x_auto = x.clone().detach()
    x_auto.requires_grad = True
    target = torch.randn((10, 5, 32, 32)).to(device)

    output_auto = loss(x_auto, target)
    output = l(x, target)

    torch.testing.assert_allclose(output, output_auto)

    output_auto.backward()
    x_grad = l.backward()
    torch.testing.assert_allclose(x_grad, x_auto.grad)

    in_channels = 8
    out_channels = 6
    kernel_size = (3, 4)
    stride = 3
    padding = 3
    dilation = 4
    bias = True
    conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding,
                           dilation=dilation, bias=bias, device=device)
    cv = Conv2d(in_channels=in_channels, out_channels=out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding,
                dilation=dilation, bias=bias)
    cv.weight = conv.weight.clone().detach()
    cv.bias = conv.bias.clone().detach()

    x = torch.randn((50, in_channels, 32, 32)).to(device)
    x_auto = x.clone().detach()
    x_auto.requires_grad = True

    output_auto = conv(x_auto)
    output_auto.retain_grad()
    output = cv(x)
    loss_auto = output_auto.sum()
    loss = output.sum()

    torch.testing.assert_allclose(output, output_auto)
    torch.testing.assert_allclose(loss, loss_auto)

    loss_auto.backward()
    output_grad = output_auto.grad.clone().detach()
    x_grad = cv.backward(output_grad)
    torch.testing.assert_allclose(x_grad, x_auto.grad)
    torch.testing.assert_allclose(cv.grad_weight, conv.weight.grad)
    torch.testing.assert_allclose(cv.grad_bias, conv.bias.grad)

    in_channels = 10
    out_channels = 10
    kernel_size = (4, 2)
    stride = 4
    padding = 2
    output_padding = 1
    dilation = 2
    bias = True
    convtrans = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, dilation=dilation,
                                         bias=bias, device=device)
    cvt = TransposeConv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=stride, padding=padding,
                          output_padding=output_padding, dilation=dilation,
                          bias=bias)
    cvt.weight = convtrans.weight.clone().detach()
    cvt.bias = convtrans.bias.clone().detach()

    x = torch.randn((3, in_channels, 32, 32)).to(device)
    x_auto = x.clone().detach()
    x_auto.requires_grad = True

    output_auto = convtrans(x_auto)
    output_auto.retain_grad()
    output = cvt(x)
    loss_auto = output_auto.sum()
    loss = output.sum()

    torch.testing.assert_allclose(output, output_auto)
    torch.testing.assert_allclose(loss, loss_auto)

    loss_auto.backward()
    output_grad = output_auto.grad.clone().detach()
    x_grad = cvt.backward(output_grad)
    torch.testing.assert_allclose(x_grad, x_auto.grad)
    torch.testing.assert_allclose(cvt.grad_weight, convtrans.weight.grad)
    torch.testing.assert_allclose(cvt.grad_bias, convtrans.bias.grad)

    nb_channels = 9
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    dilation = 1
    bias = True

    conv1 = torch.nn.Conv2d(in_channels=3, out_channels=nb_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, bias=bias, device=device)

    conv2 = torch.nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels,
                            kernel_size=kernel_size, stride=stride, padding=padding,
                            dilation=dilation, bias=bias, device=device)

    transpconv1 = torch.nn.ConvTranspose2d(in_channels=nb_channels, out_channels=nb_channels,
                                           kernel_size=kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, dilation=dilation,
                                           bias=bias, device=device)

    transpconv2 = torch.nn.ConvTranspose2d(in_channels=nb_channels, out_channels=3,
                                           kernel_size=kernel_size, stride=stride, padding=padding,
                                           output_padding=output_padding, dilation=dilation,
                                           bias=bias, device=device)

    layers = torch.nn.Sequential(conv1, torch.nn.ReLU(), conv2, torch.nn.ReLU(), transpconv1, torch.nn.ReLU(),
                                 transpconv2, torch.nn.Sigmoid())

    cv1 = Conv2d(in_channels=3, out_channels=nb_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                 dilation=dilation, bias=bias)
    cv1.weight = conv1.weight.clone().detach()
    cv1.bias = conv1.bias.clone().detach()

    cv2 = Conv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size, stride=stride,
                 padding=padding, dilation=dilation, bias=bias)
    cv2.weight = conv2.weight.clone().detach()
    cv2.bias = conv2.bias.clone().detach()

    cvt1 = TransposeConv2d(in_channels=nb_channels, out_channels=nb_channels, kernel_size=kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, dilation=dilation, bias=bias)
    cvt1.weight = transpconv1.weight.clone().detach()
    cvt1.bias = transpconv1.bias.clone().detach()

    cvt2 = TransposeConv2d(in_channels=nb_channels, out_channels=3, kernel_size=kernel_size, stride=stride,
                           padding=padding,
                           output_padding=output_padding, dilation=dilation, bias=bias)
    cvt2.weight = transpconv2.weight.clone().detach()
    cvt2.bias = transpconv2.bias.clone().detach()

    l = Sequential(cv1, ReLU(), cv2, ReLU(), cvt1, ReLU(), cvt2, Sigmoid())

    x = torch.randn((50, 3, 32, 32)).to(device)
    target = torch.randn((50, 3, 32, 32)).to(device)
    x_auto = x.clone().detach()
    x_auto.requires_grad = True

    output_auto = layers(x_auto)
    output = l(x)
    criterion_auto = torch.nn.MSELoss()
    criterion = MSE()

    loss_auto = criterion_auto(output_auto, target)
    loss = criterion(output, target)

    loss_auto.backward()
    input_grad = l.backward(criterion.backward())

    torch.testing.assert_allclose(output, output_auto)
    torch.testing.assert_allclose(loss, loss_auto)
    torch.testing.assert_allclose(input_grad, x_auto.grad)
    torch.testing.assert_allclose(cv1.grad_weight, conv1.weight.grad)
    torch.testing.assert_allclose(cv1.grad_bias, conv1.bias.grad)
    torch.testing.assert_allclose(cvt1.grad_weight, transpconv1.weight.grad)
    torch.testing.assert_allclose(cvt1.grad_bias, transpconv1.bias.grad)

    print("end")
