import Miniproject_2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10 ** -8)


def main():
    global load, save, nb_samples, num_epochs

    # Load the data
    noisy_imgs_1, noisy_imgs_2 = torch.load('./data/train_data.pkl', map_location=device)
    noisy_imgs, clean_imgs = torch.load('./data/val_data.pkl', map_location=device)
    print(noisy_imgs_1.shape)
    print(noisy_imgs_1.device)

    # Converts the data to float type
    noisy_imgs_1 = noisy_imgs_1[: nb_samples].float()
    noisy_imgs_2 = noisy_imgs_2[: nb_samples].float()
    noisy_imgs = noisy_imgs[: nb_samples].float()
    clean_imgs = clean_imgs[: nb_samples].float()

    # Normalisation of data
    noisy_imgs_1.div_(255)
    noisy_imgs_2.div_(255)
    noisy_imgs.div_(255)
    clean_imgs.div_(255)

    denoiser = Miniproject_2.Model()

    if not load:
        # Training
        denoiser.train(noisy_imgs_1, noisy_imgs_2)

    else:
        # Load the saved best model
        denoiser.load_pretrained_model()

    # Validation
    output = denoiser.predict(noisy_imgs)
    print(output.shape)
    noise_db = psnr(output, clean_imgs).item()
    print(noise_db)

    # Noise without filtering
    noise_db = psnr(noisy_imgs, clean_imgs).item()
    print(noise_db)

    if save:
        # Save the model
        torch.save(denoiser.model.state_dict(), './Miniproject_1/bestmodel.pth')


if __name__ == "__main__":
    load = False
    save = False
    nb_samples = 10
    num_epochs = 5
    main()
