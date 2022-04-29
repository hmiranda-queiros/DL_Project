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

    def train(self, train_input, train_target, num_epochs, nb_samples, noisy_imgs, clean_imgs) -> None:
        #: train_input : tensor of size (N, C, H, W) containing a noisy version of the images.
        #: train_target : tensor of size (N, C, H, W) containing another noisy version of the same images,
        # which only differs from the input by their noise .
        train_input_full = train_input.clone().detach()
        train_target_full = train_target.clone().detach()

        print(f"Starts Training with : mini_batch_size = {self.mini_batch_size} and num epochs = {num_epochs}")
        total_time = 0
        for e in range(num_epochs):
            start = time.time()
            # Shuffles data
            rand_lines = torch.randperm(train_input_full.shape[0])[:nb_samples]
            train_input = train_input_full[rand_lines]
            train_target = train_target_full[rand_lines]

            if total_time > 10 * 60:
                break

            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = self.model(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()

            end = time.time()
            total_time += end - start

            if (e + 1) % 10 == 0:
                result = self.validation(noisy_imgs, clean_imgs)

                # if result > 23 and self.lr == 0.001:
                #     self.lr = 0.001
                #     self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08)

                print(f"Epoch number : {e + 1}, PSNR : {result:.2f}, Total running time : {total_time:.1f} s")

        print(f"End of training with total running time : {total_time:.1f} s")

    def predict(self, test_input) -> torch.Tensor:
        #: test_input : tensor of size (N1 , C, H, W) that has to be denoised by the trained
        # or the loaded network.
        #: returns a tensor of the size (N1 , C, H, W)
        output = torch.empty(0).to(device)
        for b in range(0, test_input.size(0), 200):
            output = torch.cat((output, self.model(test_input.narrow(0, b, 200))), 0)
        return output

    def validation(self, noisy_imgs, clean_imgs):
        output = self.predict(noisy_imgs)
        return psnr(output, clean_imgs).item()
