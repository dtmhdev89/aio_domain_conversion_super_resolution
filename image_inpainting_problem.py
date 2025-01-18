import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet_model import II_Unet, SkipConfig, ImageInpaintingDataset, \
    LHR_TRAIN_DATA_PATH, LHR_VAL_DATA_PATH, BATCH_SIZE, \
    get_device, generate_images


if __name__ == "__main__":
    device = get_device()
    unet_model = II_Unet(
        skip_config=SkipConfig(use_skip=False)
    ).to(device)
    img = torch.ones(2, 3, 256, 256).to(device)
    print(unet_model(img).shape)

    unet_model = II_Unet(
        skip_config=SkipConfig(use_skip=True)
    ).to(device)
    img = torch.ones(2, 3, 256, 256).to(device)
    print(unet_model(img).shape)

    train_dataset = ImageInpaintingDataset(LHR_TRAIN_DATA_PATH, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True)

    test_dataset = ImageInpaintingDataset(LHR_VAL_DATA_PATH, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False)
    # in_batch, tar_batch = next(iter(train_loader))
    # in_batch = (in_batch + 1)/2
    # tar_batch = (tar_batch + 1)/2

    # plt.figure(figsize=(10, 10))
    # ax = plt.subplot(2, 2, 1)
    # plt.imshow(np.squeeze(in_batch[0].numpy().transpose((1, 2, 0))))
    # plt.title("Input")
    # ax = plt.subplot(2, 2, 3)
    # plt.imshow(np.squeeze(tar_batch[0].numpy().transpose((1, 2, 0))))
    # plt.title("Target")
    # ax = plt.subplot(2, 2, 2)
    # plt.imshow(np.squeeze(in_batch[1].numpy().transpose((1, 2, 0))))
    # plt.title("Input")
    # ax = plt.subplot(2, 2, 4)
    # plt.imshow(np.squeeze(tar_batch[1].numpy().transpose((1, 2, 0))))
    # plt.title("Target")

    # plt.show()
