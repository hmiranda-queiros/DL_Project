import torch
import time

from torch import optim
from torch import nn
from .src import denoiser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# For mini-project 1
def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10 ** -8)


class Model:
    def __init__(self, mini_batch_size) -> None:
        # instantiate model + optimizer + loss function + any other stuff you need
        self.model = denoiser.Denoiser()
        self.model.to(device)
        self.criterion = nn.MSELoss()
        self.lr = 0.001
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-08)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.mini_batch_size = mini_batch_size

    def load_pretrained_model(self) -> None:
        # This loads the parameters saved in bestmodel.pth into the model
        denoiser_state_dict = torch.load('./Miniproject_1/bestmodel.pth')
        self.model.load_state_dict(denoiser_state_dict)

    def train(self, train_input, train_target, num_epochs, test_input, test_target) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images,
        # which only differs from the input by their noise .
        print(f"Starts Training with : mini_batch_size = {self.mini_batch_size} and num epochs = {num_epochs}")
        total_time = 0
        nb_step = 0
        result = 0

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
                    result = self.validation(test_input, test_target)

                    print(
                        f"Epoch number : {e + 1}, Step number : {nb_step}, PSNR : {result:.2f},"
                        f" mini_batch_size = {self.mini_batch_size}, Total running time : {total_time:.1f} s")

                if total_time > 10 * 60:
                    print("10 mins exceeded")
                    break

            # adapt minibatch_size
            if result >= 23:
                self.mini_batch_size = 25

            if result >= 24:
                self.mini_batch_size = 25

        print(f"End of training with total running time : {total_time:.1f} s")

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)
        output = torch.empty(0).to(device)
        for b in range(0, test_input.size(0), 100):
            output = torch.cat((output, self.model(test_input.narrow(0, b, 100))), 0)
        return output

    def validation(self, test_input, test_target):
        output = self.predict(test_input)
        return psnr(output, test_target).item()
