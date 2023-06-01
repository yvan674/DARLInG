"""Fashion MNIST.

I want to check if the VAE architecture I've implemented actually works.
To do so, we will use the VAE on Fashion MNIST. This is a known dataset that
I have worked with before so it is good to use as a sanity check.

Research Questions:
    - Does the VAE actually work?

Answers:
    As of 1.6.2023 at 17:20, no it does not.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
"""
import os
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Optimizer
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

from models.encoder import BVPEncoder
from models.multi_task import MultiTaskHead
from loss.triple_loss import TripleLoss
from ui.vae_experiment_gui import TrainingGUI


def parse_args():
    """Parses arguments required for training."""
    p = ArgumentParser()
    p.add_argument("DATA_ROOT", type=Path,
                   help="Path to the FashionMNIST dataset")
    p.add_argument("CHECKPOINT_ROOT", type=Path,
                   help="Path to the checkpoints")
    p.add_argument("--debug", action="store_true",
                   help="Activates debug mode (CPU and single threaded)")
    return p.parse_args()


def tensor_to_image(data: torch.Tensor, img_idxs: tuple) -> list:
    """Takes a raw data tensor and turns it into a PIL image.

    Args:
        data: Tensor in shape [batch, 1, 32, 32].
        img_idxs: Img indices to turn into the output image.

    Returns:
        An image in mode "grayscale".
    """
    img_array = data[img_idxs[0]:img_idxs[1]]\
        .detach()\
        .cpu()\
        .numpy()\
        .reshape([-1, 28, 28])
    img_array = (img_array * 255).astype('uint8')
    img_arrays = [Image.fromarray(img) for img in img_array]

    return img_arrays


def img_to_tensor_collate_fn(batch):
    """Returns a tuple of 4d tensors over batch as dim 0 and class"""
    imgs = []
    classes = []
    for img, class_idx in batch:
        # Turn imgs into numpy arrays
        img = np.array(img)
        img = np.expand_dims(img, axis=0)  # Make 2d into 3d

        imgs.append(img)
        classes.append(class_idx)

    # Stack them into a 4d tensor
    imgs = np.array(imgs, dtype="float32") / 255.
    imgs = torch.tensor(imgs)
    classes = torch.tensor(classes)

    return imgs, classes


def train_loop(encoder: nn.Module,
               head: nn.Module,
               loss: nn.Module,
               encoder_optimizer: Optimizer,
               head_optimizer: Optimizer,
               dataloader: DataLoader,
               epochs: int,
               device: torch.device,
               checkpoint_path: Path):
    gui = TrainingGUI(0)
    gui.set_max_values(len(dataloader), epochs)
    gui.update_status("Beginning training...")
    p_bar = tqdm(total=len(dataloader) * epochs)

    step = -1
    prev_time = perf_counter()

    for epoch in range(epochs):
        gui.update_status("Running training...")
        encoder.train()
        head.train()

        for batch_idx, (imgs, label) in enumerate(dataloader):
            step += 1

            # Move to device
            imgs = imgs.to(device)
            label = label.to(device)

            # Zero gradients
            encoder_optimizer.zero_grad()
            head_optimizer.zero_grad()

            # Forward pass
            z, mu, log_sigma = encoder(None, None, imgs)
            reconst, y_pred = head(z)

            # Calculate loss
            loss_val = loss(imgs, label, mu, log_sigma, reconst, y_pred,
                            reconst, y_pred,
                            False, False)
            kl_loss, null_loss, embed_loss, joint_loss = loss_val

            joint_loss.backward()
            encoder_optimizer.step()
            head_optimizer.step()

            if batch_idx % 50 == 0:
                original_img = tensor_to_image(imgs, (0, 1))[0]
                reconstr_img = tensor_to_image(reconst, (0, 1))[0]

                gui.update_image(original_img, reconstr_img)

            p_bar.update(1)
            current_time = perf_counter()
            rate = 1 / (current_time - prev_time)
            prev_time = current_time
            gui.update_data(batch_idx + 1, epoch + 1, kl_loss, joint_loss,
                            rate)


def main(data_root: Path, checkpoint_root: Path, debug_mode: bool):
    """Sets up the main training loop."""
    if debug_mode:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.has_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # SECTION: Data Prep
    train_data = FashionMNIST(f"{data_root}", train=True, download=True)
    test_data = FashionMNIST(f"{data_root}", train=False, download=True)

    # get num CPUs available
    num_cpus = os.cpu_count()
    num_workers = 1 if debug_mode else (num_cpus - 1) // 2
    train_loader = DataLoader(train_data, num_workers=num_workers,
                              collate_fn=img_to_tensor_collate_fn,
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, num_workers=num_workers,
                             collate_fn=img_to_tensor_collate_fn,
                             batch_size=64, shuffle=True)

    # SECTION: Model initialization
    encoder = BVPEncoder(fc_input_size=8192)
    head = MultiTaskHead(decoder_ac_func=nn.ReLU,
                         decoder_dropout=0.3,
                         encoder_latent_dim=10,
                         predictor_ac_func=nn.ReLU,
                         predictor_dropout=0.3,
                         domain_label_size=0,
                         bvp_output_layers=1,
                         bvp_output_size=28,
                         num_classes=10)

    encoder.to(device)
    head.to(device)

    # SECTION: Loss and optimizers
    loss_fn = TripleLoss(1.0, 0.5)
    encoder_optimizer = SGD(encoder.parameters(), lr=1e-4)
    head_optimizer = SGD(head.parameters(), lr=1e-4)

    train_loop(encoder, head, loss_fn, encoder_optimizer, head_optimizer,
               train_loader, 10, device, checkpoint_root)


if __name__ == '__main__':
    args = parse_args()
    main(args.DATA_ROOT, args.CHECKPOINT_ROOT, args.debug)
