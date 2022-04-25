import torch

from torch import optim
from torch import nn
from .src import denoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# For mini-project 1
class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = denoiser.Denoiser()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=1e-1)
        self.mini_batch_size = 5
        pass

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        denoiser_state_dict = torch.load('./Miniproject_1/bestmodel.pth')
        self.model.load_state_dict(denoiser_state_dict)
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images,
        # which only differs from the input by their noise .
        for e in range(num_epochs):
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.model(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)
        output = torch.empty(0).to(device)
        for b in range(0, test_input.size(0), self.mini_batch_size):
            output = torch.cat((output, self.model(test_input.narrow(0, b, self.mini_batch_size))), 0)
        return output
