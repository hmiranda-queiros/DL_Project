import Miniproject_1
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(denoised, ground_truth):
    # Peak Signal to Noise Ratio : denoised and ground_truth have range [0, 1]
    mse = torch.mean((denoised - ground_truth) ** 2)
    return -10 * torch.log10(mse + 10 ** -8)


def main():
    global load, save, random, nb_samples, num_epochs

    # Load the data
    noisy_imgs_1, noisy_imgs_2 = torch.load('./data/train_data.pkl', map_location=device)
    noisy_imgs, clean_imgs = torch.load('./data/val_data.pkl', map_location=device)

    if random:
        # Selects random samples
        rand_lines = torch.randperm(noisy_imgs_1.shape[0])[:nb_samples]

        # Converts the data to float type
        noisy_imgs_1 = noisy_imgs_1[rand_lines].float()
        noisy_imgs_2 = noisy_imgs_2[rand_lines].float()

    else:
        # Converts the data to float type
        noisy_imgs_1 = noisy_imgs_1[:nb_samples].float()
        noisy_imgs_2 = noisy_imgs_2[:nb_samples].float()

    noisy_imgs = noisy_imgs.float()
    clean_imgs = clean_imgs.float()

    # Normalisation of data
    noisy_imgs_1.div_(255)
    noisy_imgs_2.div_(255)
    noisy_imgs.div_(255)
    clean_imgs.div_(255)

    denoiser = Miniproject_1.Model()

    print(f"Device used : {noisy_imgs_1.device}")
    print(f"Input shape : {noisy_imgs_1.shape}")

    if not load:
        # Training
        denoiser.train(noisy_imgs_1, noisy_imgs_2, num_epochs)

    else:
        # Load the saved best model
        denoiser.load_pretrained_model()

    # Validation
    output = denoiser.predict(noisy_imgs)
    print(f"Output shape : {output.shape}")
    noise_db = psnr(output, clean_imgs).item()
    print(f"Noise after filtering : {noise_db}")

    # Noise without filtering
    noise_db = psnr(noisy_imgs, clean_imgs).item()
    print(f"Noise without filtering : {noise_db}")

    if save:
        # Save the model
        torch.save(denoiser.model.state_dict(), './Miniproject_1/bestmodel.pth')

    # Plots the result
    images_clean = clean_imgs[:5].mul(255).to("cpu").numpy()
    images_noisy = noisy_imgs[:5].mul(255).to("cpu").numpy()
    images_filtered = output[:5].mul(255).detach().to("cpu").numpy()

    fig = plt.figure(figsize=(25, 4))
    for i in range(3):
        for idx in range(5):
            fig.add_subplot(3, 5, idx + 1 + 5 * i, xticks=[], yticks=[])

            if i == 0:
                plt.imshow(np.transpose(images_clean[idx], (1, 2, 0)).astype('uint8'))

            elif i == 1:
                plt.imshow(np.transpose(images_noisy[idx], (1, 2, 0)).astype('uint8'))

            else:
                plt.imshow(np.transpose(images_filtered[idx], (1, 2, 0)).astype('uint8'))
    plt.show()


if __name__ == "__main__":
    load = False
    save = False
    random = True
    nb_samples = 50000
    num_epochs = 1
    main()