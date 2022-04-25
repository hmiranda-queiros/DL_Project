import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Module(object):
    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []
