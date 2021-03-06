import Miniproject_1
import torch
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def psnr(x, y, max_range=1.0):
    assert x.shape == y.shape and x.ndim == 4
    return 20 * torch.log10(torch.tensor(max_range)) - 10 * torch.log10(((x - y) ** 2).mean((1, 2, 3))).mean()


def main():
    global load, save, num_epochs, mini_batch_size

    # Load the data
    noisy_imgs_1, noisy_imgs_2 = torch.load('./data/train_data.pkl', map_location=device)
    noisy_imgs, clean_imgs = torch.load('./data/val_data.pkl', map_location=device)

    # Converts the data to float type
    noisy_imgs_1 = noisy_imgs_1.float()
    noisy_imgs_2 = noisy_imgs_2.float()

    noisy_imgs = noisy_imgs.float()
    clean_imgs = clean_imgs.float()

    # Normalisation of data
    noisy_imgs_1.div_(255)
    noisy_imgs_2.div_(255)
    noisy_imgs.div_(255)
    clean_imgs.div_(255)

    denoiser = Miniproject_1.Model(mini_batch_size)

    print(f"Device used : {noisy_imgs_1.device}")
    print(f"Input shape : {noisy_imgs_1.shape}")

    if load:
        # Load the saved best model
        denoiser.load_pretrained_model()

    else:
        output = denoiser.predict(noisy_imgs)
        noise_db = psnr(output, clean_imgs).item()
        print(f"PSNR at step 0 of training : {noise_db:.2f}")
        # Training
        denoiser.train(noisy_imgs_1, noisy_imgs_2, num_epochs, noisy_imgs, clean_imgs)

    # Validation
    print("Starts Validation")
    output = denoiser.predict(noisy_imgs)
    print(f"Output shape : {output.shape}")
    noise_db = psnr(output, clean_imgs).item()
    print(f"Noise after denoising : {noise_db:.2f}")

    # Noise without denoising
    noise_db = psnr(noisy_imgs, clean_imgs).item()
    print(f"Noise without denoising : {noise_db:.2f}")

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
    load = True
    save = False
    num_epochs = 1000
    mini_batch_size = 50
    main()
