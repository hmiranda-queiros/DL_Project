import torch
import time

from torch import optim
from torch import nn
from .others import denoiser

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Model:
    def __init__(self) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = denoiser.Denoiser()
        self.model.to(device)
        self.criterion = nn.MSELoss()
        self.lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08)
        self.mini_batch_size = 50

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        denoiser_state_dict = torch.load('./Miniproject_1/bestmodel.pth')
        self.model.load_state_dict(denoiser_state_dict)

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
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                self.model.zero_grad()
                loss.backward()
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
