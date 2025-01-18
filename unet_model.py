import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchsummary import summary
from torchvision import transforms
import os
import matplotlib.pyplot as plt
import numpy as np
from torcheval.metrics.functional import peak_signal_noise_ratio
import time
import datetime
import cv2

torch.manual_seed(66)
print('set torch seed')

def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


class SkipConfig(nn.Module):
    def __init__(self, use_skip: bool = True) -> None:
        super(SkipConfig, self).__init__()
        self.use_skip = use_skip

    def forward(self, x, skip):
        if not self.use_skip:
            return x
        else:
            return torch.cat([x, skip], dim=1)


class FirstFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class BaseBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: tuple = (3, 3)) -> None:
        super(BaseBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=self.kernel_size,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 base_block=BaseBlock):
        super(ConvBlock, self).__init__()
        self.base_block = base_block
        self.conv = nn.Sequential(
            self.base_block(in_channels,
                            out_channels,
                            kernel_size=(3, 3)),
            self.base_block(out_channels,
                            out_channels,
                            kernel_size=(3, 3))
        )

    def forward(self, x):
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2) -> None:
        super(UpsamplingBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=(1, 1),
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_config=SkipConfig(use_skip=True)):
        super(Decoder, self).__init__()
        self.skip_config = skip_config
        skip_channels = out_channels
        if not skip_config.use_skip:
            skip_channels = out_channels * 2
        self.conv = UpsamplingBlock(in_channels, skip_channels, scale_factor=2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = self.skip_config(x, skip)
        x = self.conv_block(x)
        return x


class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(
            self,
            n_channels=3,
            n_classes=3,
            skip_config=SkipConfig(use_skip=True),
    ):
        super(Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = ConvBlock(64, 64)

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)

        self.dec_1 = Decoder(1024, 512, skip_config=skip_config)
        self.dec_2 = Decoder(512, 256, skip_config=skip_config)
        self.dec_3 = Decoder(256, 128, skip_config=skip_config)
        self.dec_4 = Decoder(128, 64, skip_config=skip_config)

        self.out_conv = FinalOutput(64, n_classes)

    def forward(self, x):
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        x = self.out_conv(x)

        return x


class ImageDataset(Dataset):
    LOW_IMG_HEIGHT = 64
    LOW_IMG_WIDTH = 64

    def __init__(self, img_dir, is_train=True):
        image_shape = (self.LOW_IMG_WIDTH, self.LOW_IMG_HEIGHT)
        self.resize = transforms.Resize(image_shape, antialias=True)
        self.is_train = is_train
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def normalize(self, input_image, target_image):
        input_image = input_image * 2 - 1
        target_image = target_image * 2 - 1

        return input_image, target_image

    def random_jitter(self, input_image, target_image):
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)

        return input_image, target_image

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        image = transforms.functional.to_tensor(image)
        input_image = self.resize(image).type(torch.float32)
        target_image = image.type(torch.float32)
        input_image, target_image = self.normalize(input_image=input_image,
                                                   target_image=target_image)
        if self.is_train:
            input_image, target_image = self.random_jitter(
                input_image=input_image,
                target_image=target_image
            )

        return input_image, target_image


class ImageInpaintingDataset(Dataset):
    IMG_HEIGHT = 256
    IMG_WIDTH = 256

    def __init__(self, img_dir, is_train=True):
        self.is_train = is_train
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def normalize(self, input_image, target_image):
        input_image = input_image * 2 - 1
        target_image = target_image * 2 - 1

        return input_image, target_image

    def random_jitter(self, input_image, target_image):
        if torch.rand([]) < 0.5:
            input_image = transforms.functional.hflip(input_image)
            target_image = transforms.functional.hflip(target_image)

        return input_image, target_image

    def create_mask(self, image):
        masked_image = image.copy()
        # Prepare masking matrix
        mask = np.full((self.IMG_WIDTH, self.IMG_HEIGHT, 3), 0, np.uint8)

        for _ in range(np.random.randint(1, 5)):
            # Get random x locations to start line
            x1, x2 = (
                np.random.randint(1, self.IMG_WIDTH),
                np.random.randint(1, self.IMG_WIDTH)
            )
            # Get random y locations to start line
            y1, y2 = (
                np.random.randint(1, self.IMG_HEIGHT),
                np.random.randint(1, self.IMG_HEIGHT)
            )
            # Get random thickness of the line drawn
            thickness = np.random.randint(1, 15)
            # Draw line on the black mask
            cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

        masked_image = np.where(mask, 255*np.ones_like(mask), masked_image)

        return masked_image

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))

        input_image = self.create_mask(image)
        input_image = transforms.functional.to_tensor(input_image)
        target_image = transforms.functional.to_tensor(image)

        input_image = input_image.type(torch.float32)
        target_image = target_image.type(torch.float32)

        input_image, target_image = self.normalize(input_image=input_image,
                                                   target_image=target_image)
        if self.is_train:
            input_image, target_image = self.random_jitter(
                input_image=input_image,
                target_image=target_image
            )

        return input_image, target_image


class SR_Unet(nn.Module):
    LOW_IMG_HEIGHT = 64
    LOW_IMG_WIDTH = 64

    def __init__(self,
                 n_channels=3,
                 n_classes=3,
                 skip_config=SkipConfig(use_skip=True)) -> None:
        super(SR_Unet, self).__init__()
        resize_shape = (self.LOW_IMG_WIDTH * 4, self.LOW_IMG_HEIGHT * 4)
        self.resize_fn = transforms.Resize(resize_shape, antialias=True)
        self.unet = Unet(
            n_channels=n_channels,
            n_classes=n_classes,
            skip_config=skip_config)

    def forward(self, x):
        x = self.resize_fn(x)
        x = self.unet(x)

        return x


class II_Unet(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=3,
                 skip_config=SkipConfig(use_skip=True)) -> None:
        super(II_Unet, self).__init__()
        self.unet = Unet(
            n_channels=n_channels,
            n_classes=n_classes,
            skip_config=skip_config)

    def forward(self, x):
        x = self.unet(x)

        return x


def generate_images(model, inputs, labels, device='cpu'):
    model.eval()
    with torch.no_grad():
        inputs, labels = inputs.to(device), labels.to(device)
        predictions = model(inputs)

    inputs, labels, predictions = [
                                    inputs.cpu().numpy(),
                                    labels.cpu().numpy(),
                                    predictions.cpu().numpy()
                                  ]
    plt.figure(figsize=(15, 20))

    display_list = [
        inputs[-1].transpose((1, 2, 0)),
        labels[-1].transpose((1, 2, 0)),
        predictions[-1].transpose((1, 2, 0))
    ]
    title = ['Input', 'Real', 'Predicted']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow((display_list[i] + 1) / 2)
        plt.axis('off')

    try:
        from IPython.display import Image as ipython_image
    except ImportError:
        print("Not running in Jupyter Notebook. Saving image.")
        timestamp_format = "%Y%m%d_%H%M%S_%f"
        timestamp = datetime.datetime.now().strftime(timestamp_format)
        plt.savefig(f'predicted_{timestamp}.png')
        plt.close()
    else:
        plt.show()


def train_epoch(model, optimizer, criterion, train_dataloader, device, epoch=0,
                log_interval=50):
    model.train()
    total_psnr, total_count = 0, 0
    losses = []
    # start_time = time.time()

    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)

        # compute loss
        loss = criterion(predictions, labels)
        losses.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()

        total_psnr += peak_signal_noise_ratio(predictions, labels)
        total_count += 1
        if idx % log_interval == 0 and idx > 0:
            # elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches "
                "| psnr {:8.3f}".format(
                    epoch, idx, len(train_dataloader), total_psnr / total_count
                )
            )
            total_psnr, total_count = 0, 0
            # start_time = time.time()

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss


def evaluate_epoch(model, criterion, valid_dataloader, device):
    model.eval()
    total_psnr, total_count = 0, 0
    losses = []

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            predictions = model(inputs)

            loss = criterion(predictions, labels)
            losses.append(loss.item())

            total_psnr += peak_signal_noise_ratio(predictions, labels)
            total_count += 1

    epoch_psnr = total_psnr / total_count
    epoch_loss = sum(losses) / len(losses)
    return epoch_psnr, epoch_loss


def train(model, model_name, save_model, optimizer,
          criterion, train_dataloader, valid_dataloader,
          num_epochs, device):
    train_psnrs, train_losses = [], []
    eval_psnrs, eval_losses = [], []
    best_psnr_eval = -1000
    times = []
    for epoch in range(1, num_epochs+1):
        epoch_start_time = time.time()
        # Training
        train_psnr, train_loss = train_epoch(model, optimizer, criterion,
                                             train_dataloader, device, epoch)
        train_psnrs.append(train_psnr.cpu())
        train_losses.append(train_loss)

        # Evaluation
        eval_psnr, eval_loss = evaluate_epoch(model, criterion,
                                              valid_dataloader, device)
        eval_psnrs.append(eval_psnr.cpu())
        eval_losses.append(eval_loss)

        # Save best model
        if best_psnr_eval < eval_psnr:
            torch.save(model.state_dict(), save_model + f'/{model_name}.pt')
            inputs_t, targets_t = next(iter(valid_dataloader))
            generate_images(model, inputs_t, targets_t, device=device)
            best_psnr_eval = eval_psnr
        times.append(time.time() - epoch_start_time)
        # Print loss, psnr end epoch
        print("-" * 59)
        print(
            "| End of epoch {:3d} | Time: {:5.2f}s "
            "| Train psnr {:8.3f} | Train Loss {:8.3f} "
            "| Valid psnr {:8.3f} | Valid Loss {:8.3f} ".format(
                epoch, time.time() - epoch_start_time,
                train_psnr, train_loss,
                eval_psnr, eval_loss
            )
        )
        print("-" * 59)

    # Load best model
    # Not a secure way to load model with a pretrained weights. Use safetensors instead
    model.load_state_dict(torch.load(save_model + f'/{model_name}.pt'))
    model.eval()
    metrics = {
        'train_psnr': train_psnrs,
        'train_loss': train_losses,
        'valid_psnr': eval_psnrs,
        'valid_loss': eval_losses,
        'time': times
    }
    return model, metrics


def plot_result(num_epochs, train_psnrs,
                eval_psnrs, train_losses, eval_losses):
    epochs = list(range(num_epochs))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs[0].plot(epochs, train_psnrs, label="Training")
    axs[0].plot(epochs, eval_psnrs, label="Evaluation")
    axs[1].plot(epochs, train_losses, label="Training")
    axs[1].plot(epochs, eval_losses, label="Evaluation")
    axs[0].set_xlabel("Epochs")
    axs[1].set_xlabel("Epochs")
    axs[0].set_ylabel("PSNR")
    axs[1].set_ylabel("Loss")
    plt.legend()
    try:
        from IPython.display import Image as ipython_image
    except ImportError:
        print("Not running in Jupyter Notebook. Saving image.")
        timestamp_format = "%Y%m%d_%H%M%S_%f"
        timestamp = datetime.datetime.now().strftime(timestamp_format)
        plt.savefig(f'predicted_{timestamp}.png')
        plt.close()
    else:
        plt.show()


def predict_and_display(model, test_dataloader, device):
    model.eval()

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_dataloader):
            if idx >= 10:
                break
            inputs = inputs.to(device)
            model(inputs)
            generate_images(model, inputs, labels, device=device)


# constant
LHR_TRAIN_DATA_PATH = os.path.join('Khoa_LHR_image/train')
LHR_VAL_DATA_PATH = os.path.join('Khoa_LHR_image/val')
BATCH_SIZE = 8


if __name__ == "__main__":
    print('This is base model')
