"""Train.

Training function for DARLInG.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from time import perf_counter

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, F1Score, \
    ConfusionMatrix
import wandb
from wandb.wandb_run import Run

from ui.base_ui import BaseUI
from models.base_embedding_agent import BaseEmbeddingAgent


def train(encoder: nn.Module,
          null_multitask_head: nn.Module,
          embed_multitask_head: nn.Module,
          embedding_agent: BaseEmbeddingAgent,
          train_embedding_agent: bool,
          null_embedding: torch.Tensor,
          encoder_optimizer: Optimizer,
          null_multitask_optimizer: Optimizer,
          embed_multitask_optimizer: Optimizer,
          kl_loss: nn.Module,
          mt_loss: nn.Module,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          epochs: int,
          logging: Run,
          device: torch.device,
          chekckpoint_path: Path,
          ui: BaseUI):
    """Performs training on DARLInG.

    Args:
        encoder: The CNN encoder which encodes the imaged time series data.
        null_multitask_head: The MT head which receives the null domain
            embedding.
        embed_multitask_head: The MT head which recieves some non-null domain
            embedding.
        embedding_agent: The agent which performs the domain embedding.
        train_embedding_agent: Whether to train the embedding agent. If the
            embedding agent is not using RL, for example, don't train it.
        null_embedding: The value to use for the null embedding.
        encoder_optimizer: Optimizer for the CNN encoder.
        null_multitask_optimizer: Optimizer for the null domain MT head.
        embed_multitask_optimizer: Optimizer for the non-null domain MT head.
        kl_loss: ELBO loss for the encoder.
        mt_loss: Loss function for the MT heads.
        train_loader: The training data loader.
        valid_loader: The validation data loader.
        epochs: The number of epochs to train for.
        logging: The logger to use for logging.
        device: The device to use for training.
        chekckpoint_path: The path to save checkpoints to.
        ui: The UI to use to visualize training.
    """
    # This is hard coded here since we have decided to only identify
    # 6 gestures.
    num_classes = 6

    # We want to keep track of previous values to check whether we should save
    # a checkpoint after validation
    best_joint_loss = 1.0e8
    prev_checkpoint_fp: Path = chekckpoint_path / "this-doesnt-exist.pth"
    # In case any models haven't been sent to device yet
    encoder.to(device)
    null_multitask_head.to(device)
    embed_multitask_head.to(device)

    acc = Accuracy(task="multiclass", num_classes=num_classes)
    prec = Precision(task="multiclass", num_classes=num_classes)
    f1 = F1Score(task="multiclass", num_classes=num_classes)

    def forward_pass(a, phi, b):
        """Runs a single forward pass."""
        a, phi, b = a.to(device), phi.to(device), b.to(device)

        # Forward pass
        z_pred, mu_pred, log_sigma_pred = encoder(a, phi, b)

        # Generate domain embeddings
        domain_embedding = embedding_agent(z_pred)
        batch_null_embedding = torch.cat(len(a) * [null_embedding])

        # Run the multitask heads
        b_null, y_pred_null = null_multitask_head(
            torch.cat([z_pred, batch_null_embedding], dim=1)
        )
        b_embed, y_pred_embed = embed_multitask_head(
            torch.cat([z_pred, domain_embedding], dim=1)
        )
        return z, mu_pred, log_sigma_pred, b_null, y_pred_null, b_embed, \
            y_pred_embed

    def calculate_losses(bvp_gt, gesture_gt, mu_pred, log_sigma_pred,
                         bvp_null_pred, gesture_null_pred,
                         bvp_embed_pred, gesture_embed_pred,
                         reconstr_loss_only, no_kl_loss):
        if not no_kl_loss:
            kl_loss_val = kl_loss(mu_pred, log_sigma_pred)
        else:
            kl_loss_val = None
        null_loss_val = mt_loss(bvp_gt, gesture_gt, bvp_null_pred,
                                gesture_null_pred,
                                reconstr_loss_only)
        embed_loss_val = mt_loss(bvp_gt, gesture_gt, bvp_embed_pred,
                                 gesture_embed_pred,
                                 reconstr_loss_only)

        return kl_loss_val, null_loss_val, embed_loss_val

    step = -1

    for epoch in range(epochs):
        encoder.train()
        null_multitask_head.train()
        embed_multitask_head.train()
        embedding_agent.eval()

        # SECTION: Train the VAE classifier
        for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
            step += 1

            start_time = perf_counter()
            gesture = info["gesture"].to(device)

            _, mu, log_sigma, bvp_null, gesture_null, bvp_embed, \
                gesture_embed = forward_pass(amp, phase, bvp)

            # Calculate losses
            kl_loss_value, null_loss_value, embed_loss_value = calculate_losses(
                bvp, gesture, mu, log_sigma, bvp_null, gesture_null,
                bvp_embed, gesture_embed, False, False
            )

            # Backward pass
            encoder_optimizer.zero_grad()
            null_multitask_optimizer.zero_grad()
            embed_multitask_optimizer.zero_grad()
            kl_loss_value.backward()
            null_loss_value.backward()
            embed_loss_value.backward()
            encoder_optimizer.step()
            null_multitask_optimizer.step()
            embed_multitask_optimizer.step()

            # Calculate metrics
            kl_loss_value = kl_loss_value.data
            null_loss_value = null_loss_value.data
            embed_loss_value = embed_loss_value.data
            joint_loss_value = (kl_loss_value
                                + null_loss_value
                                + embed_loss_value)
            log_dict = {
                "train_loss": joint_loss_value,
                "train_kl_loss": kl_loss_value,
                "train_null_loss": null_loss_value,
                "train_embed_loss": embed_loss_value,
                "train_mus": wandb.Histogram(mu.mean(dim=0)
                                             .detach()
                                             .cpu()),
                "train_log_sigmas": wandb.Histogram(log_sigma.mean(dim=0)
                                                    .detach()
                                                    .cpu())
            }

            # Add images every 50 batches.
            if batch_idx % 50 == 0:
                visualize_and_store_images(bvp,
                                           bvp_null,
                                           bvp_embed,
                                           log_dict,
                                           "train",
                                           ui)

            current_time = perf_counter()
            rate = 1 / (current_time - start_time)

            ui.update_data({"train_loss": joint_loss_value,
                            "train_kl_loss": kl_loss_value,
                            "train_null_loss": null_loss_value,
                            "train_embed_loss": embed_loss_value,
                            "loss_diff": embed_loss_value - null_loss_value,
                            "epoch": epoch,
                            "batch": batch_idx,
                            "rate": rate})
            logging.log(log_dict, step)
            ui.step(1)

        # SECTION: Freeze the models, unfreeze the embedding agent
        encoder.eval()
        null_multitask_head.eval()
        embed_multitask_head.eval()
        embedding_agent.train()

        # Train the embedding agent
        if train_embedding_agent:
            ui.update_status("Training embedding agent...")
            for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
                start_time = perf_counter()
                gesture = info["gesture"].to(device)

                z, mu, log_sigma, bvp_null, gesture_null, bvp_embed, \
                    gesture_embed = forward_pass(amp, phase, bvp)

                _, null_loss_value, embed_loss_value = calculate_losses(
                    bvp, gesture, mu, log_sigma, bvp_null, gesture_null,
                    bvp_embed, gesture_embed, False, True
                )
                reward = embed_loss_value.data - null_loss_value.data

                embedding_agent.process_reward(z, reward)


        # Freeze everything for validation
        embedding_agent.eval()

        # SECTION: Perform Validation
        kl_losses = []
        joint_losses = []
        bvp_null_losses = []
        bvp_embed_losses = []
        gesture_gts = []
        gesture_null_preds = []
        gesture_embed_preds = []
        bvp = None
        bvp_null = None
        bvp_embed = None

        ui.update_status("Running validation...")
        for batch_idx, (amp, phase, bvp, info) in enumerate(valid_loader):
            start_time = perf_counter()
            gesture = info["gesture"].to(device)

            _, mu, log_sigma, bvp_null, gesture_null, bvp_embed, \
                gesture_embed = forward_pass(amp, phase, bvp)

            # Calculate losses
            kl_loss_value, null_loss_value, embed_loss_value = calculate_losses(
                bvp, gesture,
                mu, log_sigma,
                bvp_null, gesture_null,
                bvp_embed, gesture_embed,
                reconstr_loss_only=True,
                no_kl_loss=False
            )
            # Extract data only from the losses
            kl_loss_value, null_loss_value, embed_loss_value = [
                i.data
                for i in (kl_loss_value, null_loss_value, embed_loss_value)
            ]
            joint_loss_value = (kl_loss_value
                                + null_loss_value
                                + embed_loss_value)

            # Add stuff to lists
            kl_losses.append(kl_loss_value)
            joint_losses.append(joint_loss_value)
            bvp_null_losses.append(null_loss_value)
            bvp_embed_losses.append(embed_loss_value)
            gesture_gts.append(gesture.detach())
            gesture_null_preds.append(gesture_null)
            gesture_embed_preds.append(gesture_embed)

            current_time = perf_counter()
            rate = 1 / (current_time - start_time)
            ui.update_data({"valid_loss": joint_loss_value,
                            "loss_diff": embed_loss_value - null_loss_value,
                            "epoch": epoch,
                            "batch": batch_idx,
                            "rate": rate})


        gesture_gts = torch.tensor(gesture_gts)
        gesture_null_preds = torch.tensor(gesture_null_preds)
        gesture_embed_preds = torch.tensor(gesture_embed_preds)

        # Calculate metrics over entire validation set
        joint_losses = np.mean(np.array(joint_losses))
        log_dict = {
            "valid_kl_loss": np.mean(np.array(kl_losses)),
            "valid_joint_loss": joint_losses,
            "valid_bvp_null_loss": np.mean(np.array(bvp_null_losses)),
            "valid_bvp_embed_loss": np.mean(np.array(bvp_embed_losses)),
            "valid_null_acc": acc(gesture_gts, gesture_null_preds),
            "valid_null_prec": prec(gesture_gts, gesture_null_preds),
            "valid_null_f1": f1(gesture_gts, gesture_null_preds),
            "valid_null_conf_mat": conf_matrix(gesture_gts, gesture_null_preds,
                                               num_classes),
            "valid_embed_acc": acc(gesture_gts, gesture_embed_preds),
            "valid_embed_prec": prec(gesture_gts, gesture_embed_preds),
            "valid_embed_f1": f1(gesture_gts, gesture_embed_preds),
            "valid_embed_conf_mat": conf_matrix(gesture_gts,
                                                gesture_embed_preds,
                                                num_classes)
        }

        visualize_and_store_images(bvp, bvp_null, bvp_embed, log_dict,
                                   "valid", ui)

        logging.log(log_dict, step=step)

        if joint_losses < best_joint_loss:
            checkpoint_fp = (chekckpoint_path
                             / f"{logging.name}-ep-{epoch}.pth")
            if prev_checkpoint_fp.exists():
                prev_checkpoint_fp.unlink()

            torch.save(
                {"encoder_state_dict": encoder.state_dict(),
                 "null_mt_head_state_dict": null_multitask_head.state_dict(),
                 "embed_mt_head_state_dict": embed_multitask_head.state_dict(),
                 "embed_agent_state_dict": embedding_agent.state_dict()},
                checkpoint_fp
            )

            best_joint_loss = joint_losses
            prev_checkpoint_fp = checkpoint_fp

        # Cleanup
        del kl_losses
        del joint_losses
        del bvp_null_losses
        del bvp_embed_losses
        del gesture_gts
        del gesture_null_preds
        del gesture_embed_preds
        del bvp
        del bvp_null
        del bvp_embed


def tensor_to_image(data: torch.Tensor, img_idxs: tuple) -> list:
    """Takes a raw data tensor and turns it into a PIL image.

    Args:
        data: Tensor in shape [batch, 1, 32, 32].
        img_idxs: Img indices to turn into the output image.

    Returns:
        An image in mode "grayscale".
    """
    img_array = data[img_idxs[0]:img_idxs[1]] \
        .detach() \
        .cpu() \
        .numpy() \
        .reshape([-1, 32, 32])
    img_array = (img_array * 255).astype('uint8')
    img_arrays = [Image.fromarray(img) for img in img_array]

    return img_arrays


def visualize_and_store_images(bvp: torch.Tensor,
                               bvp_null: torch.Tensor,
                               bvp_embed: torch.Tensor,
                               log_dict: dict[str, any],
                               log_prefix: str,
                               ui: BaseUI):
    """Reused in training and validation code.

    Args:
        bvp: Ground truth BVP tensor.
        bvp_null: BVP predicted by null MT head.
        bvp_embed: BVP predicted by embed MT head.
        log_dict: The log dictionary currently being used
        log_prefix: Entry prefix to add in the log dict. Usually `train` or
            `valid`.
        ui: The UI object currently being used.
    """
    original_imgs = tensor_to_image(bvp, (0, 3))
    null_reconstr_imgs = tensor_to_image(bvp_null, (0, 3))
    embed_reconstr_imgs = tensor_to_image(bvp_embed, (0, 3))

    log_dict[f"{log_prefix}_bvp"] = [wandb.Image(i)
                                     for i in original_imgs]
    log_dict[f"{log_prefix}_bvp_null"] = [wandb.Image(i)
                                          for i in null_reconstr_imgs]
    log_dict[f"{log_prefix}_bvp_embed"] = [wandb.Image(i)
                                           for i in embed_reconstr_imgs]

    ui.update_image(original_imgs[0],
                    null_reconstr_imgs[0],
                    embed_reconstr_imgs[0])


def conf_matrix(gesture_gt, gesture_pred, num_classes):
    """Generates a conf-matrix plot"""
    conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
    fig, ax = plt.subplot()
    ax.matshow(conf_mat(gesture_gt, gesture_pred))

    return fig
