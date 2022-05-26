import torch
import time
import pickle

from torch import empty
from torch.nn.functional import fold, unfold

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        from .others import denoiser

        self.model = denoiser.Denoiser()
        self.criterion = MSE()
        self.lr = 0.01
        self.optimizer = SGD(self.model.param(), lr=self.lr)
        self.mini_batch_size = 50

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        filename = './Miniproject_2/bestmodel.pth'
        infile = open(filename, 'rb')
        parameters = pickle.load(infile)
        infile.close()

        # send to correct device
        for i in range(len(parameters)):
            if parameters[i]:
                parameters[i][0][0] = parameters[i][0][0].to(device)
                parameters[i][1][0] = parameters[i][1][0].to(device)
        self.model.load(parameters)

    def train(self, train_input, train_target, num_epochs) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images,
        # which only differs from the input by their noise .

        # Normalisation of data
        train_input = train_input.to(device)
        train_input = train_input.float() / 255.0
        train_target = train_target.to(device)
        train_target = train_target.float() / 255.0

        print(f"Starts Training with : mini_batch_size = {self.mini_batch_size} and num epochs = {num_epochs}")
        total_time = 0
        nb_step = 0

        for e in range(num_epochs):
            for b in range(0, train_input.size(0), self.mini_batch_size):
                start = time.time()

                output = self.model(train_input.narrow(0, b, self.mini_batch_size))
                self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                self.model.zero_grad()
                self.model.backward(self.criterion.backward())
                self.optimizer.step()
                nb_step += 1

                end = time.time()
                total_time += end - start

                if nb_step % 100 == 0:
                    print(
                        f"Epoch number : {e + 1}, Step number : {nb_step},"
                        f" mini_batch_size = {self.mini_batch_size}, Total running time : {total_time:.1f} s")

        print(f"End of training with total running time : {total_time:.1f} s")

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)
        # Normalisation of data
        test_input = test_input.to(device)
        test_input = test_input.float() / 255.0

        output = self.model(test_input)
        output = output * 255.0
        return output


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
