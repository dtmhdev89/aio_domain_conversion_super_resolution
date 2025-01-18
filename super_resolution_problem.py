import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from torcheval.metrics.functional import peak_signal_noise_ratio
from unet_model import BATCH_SIZE, SR_Unet, SkipConfig, LHR_TRAIN_DATA_PATH, \
    LHR_VAL_DATA_PATH, ImageDataset, train, plot_result, evaluate_epoch, \
    predict_and_display


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # SR Unet without skip connection
    unet_model = SR_Unet(skip_config=SkipConfig(use_skip=False)).to(device)
    img = torch.ones(2, 3, 64, 64).to(device)
    print(unet_model(img).shape)

    # SR Unet with skip connection
    unet_model = SR_Unet(skip_config=SkipConfig(use_skip=True)).to(device)
    img = torch.ones(2, 3, 64, 64).to(device)
    print(unet_model(img).shape)

    train_dataset = ImageDataset(LHR_TRAIN_DATA_PATH, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = ImageDataset(LHR_VAL_DATA_PATH, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    in_batch, tar_batch = next(iter(train_loader))
    in_batch = (in_batch + 1)/2
    tar_batch = (tar_batch + 1)/2

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(2, 2, 1)
    plt.imshow(np.squeeze(in_batch[0].numpy().transpose((1, 2, 0))))
    plt.title("Input")
    ax = plt.subplot(2, 2, 3)
    plt.imshow(np.squeeze(tar_batch[0].numpy().transpose((1, 2, 0))))
    plt.title("Target")
    ax = plt.subplot(2, 2, 2)
    plt.imshow(np.squeeze(in_batch[1].numpy().transpose((1, 2, 0))))
    plt.title("Input")
    ax = plt.subplot(2, 2, 4)
    plt.imshow(np.squeeze(tar_batch[1].numpy().transpose((1, 2, 0))))
    plt.title("Target")
    plt.show()

    SR_unet_model_noskip = SR_Unet(
        skip_config=SkipConfig(use_skip=False)
    ).to(device)
    SR_unet_model_noskip.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(SR_unet_model_noskip.parameters(),
                           lr=1e-4, betas=(0.5, 0.999))
    save_model = './UNET'
    os.makedirs(save_model, exist_ok=True)

    EPOCHS = 100
    SR_unet_model_noskip, metrics = train(
        SR_unet_model_noskip,
        'SR_unet_model_noskip',
        save_model,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        EPOCHS,
        device
    )

    plot_result(
        EPOCHS,
        metrics["train_psnr"],
        metrics["valid_psnr"],
        metrics["train_loss"],
        metrics["valid_loss"]
    )

    test_psnr, test_loss = evaluate_epoch(SR_unet_model_noskip, criterion,
                                          test_loader, device)
    print(f'SR UNET NO SKIP:\t {test_psnr} \t {test_loss}')

    predict_and_display(SR_unet_model_noskip, train_loader, device)

    # Unet with skip connection
    SR_unet_model = SR_Unet(skip_config=SkipConfig(use_skip=True)).to(device)
    SR_unet_model.to(device)

    criterion = nn.L1Loss()

    optimizer = optim.Adam(SR_unet_model.parameters(),
                           lr=1e-4,
                           betas=(0.5, 0.999))

    save_model = './UNET'
    os.makedirs(save_model, exist_ok=True)

    EPOCHS = 100
    SR_unet_model, metrics = train(
        SR_unet_model,
        'SR_unet_model',
        save_model,
        optimizer,
        criterion,
        train_loader,
        test_loader,
        EPOCHS,
        device
    )

    plot_result(
        EPOCHS,
        metrics["train_psnr"],
        metrics["valid_psnr"],
        metrics["train_loss"],
        metrics["valid_loss"]
    )

    test_psnr, test_loss = evaluate_epoch(SR_unet_model, criterion,
                                          test_loader, device)
    print(f'SR Unet with skip connection: \t {test_psnr}, {test_loss}')
    predict_and_display(SR_unet_model, train_loader, device)
