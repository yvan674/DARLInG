"""Train.

Training function for DARLInG.
"""
import sys
from pathlib import Path
from PIL import Image
from time import perf_counter

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import wandb
from wandb.wandb_run import Run

from ui.base_ui import BaseUI


def train_old(model: nn.Module,
          loss_fn: nn.Module,
          optimizer: Optimizer,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          epochs: int,
          logging_freq: int,
          logging: Run,
          device: torch.device,
          checkpoint_path: Path,
          ui: any = None):
    """Training loop with included validation at the end of each epoch.

    Args:
        model: The model to be trained.
        loss_fn: The loss function used to train the model.
        optimizer: The optimizer used
        train_loader: The DataLoader object containing the training d.
        valid_loader: The DataLoader object containing the validation d.
        epochs: The number of epochs to train for.
        logging_freq: How often to log.
        logging: Run from wandb to log to or the basic logger.
        device: The device training should be done on.
        checkpoint_path: Path to save checkpoints to.
        ui: A Training UI, if desired.
    """
    model.to(device)  # In case it hasn't been sent to device yet
    p_bar = tqdm(total=len(train_loader) * epochs)

    logging.watch(model, log="all", log_freq=logging_freq, log_graph=True)

    ce_metric = nn.CrossEntropyLoss()

    step = -1
    if ui:
        start_time = perf_counter()
    prev_ood_metric = 0.
    for epoch in range(epochs):
        # Training
        model.train()
        if ui:
            ui.update_status(f"Running training...")
        for batch_idx, (data, target) in enumerate(train_loader):
            step += 1

            # turn from 5d array to 4d array by flattening the first two
            # dimensions
            data = data.reshape(-1, 1, 32, 32)
            target = target.reshape(-1)
            data, target = data.to(device), target.to(device)

            # Forward pass
            reconstr_x, class_preds, mus, log_sigmas = model(data)
            elbo_loss, class_loss, joint_loss = loss_fn(
                data, reconstr_x, class_preds, target, mus, log_sigmas
            )

            # Backward pass
            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()

            # Calculate metrics
            log_dict = {
                "train_loss": joint_loss.data,
                "train_elbo_loss": elbo_loss.data,
                "train_class_loss": class_loss.data,
                "train_mus": wandb.Histogram(mus.mean(dim=0)
                                             .detach()
                                             .cpu()),
                "train_log_sigmas": wandb.Histogram(log_sigmas.mean(dim=0)
                                                    .detach()
                                                    .cpu())
            }
            if batch_idx % 50 == 0:
                original_imgs = tensor_to_image(data, (0, 3))
                reconstr_imgs = tensor_to_image(reconstr_x, (0, 3))

                log_dict["train_inputs"] = [wandb.Image(i)
                                            for i in original_imgs]
                log_dict["train_reconstructions"] = [wandb.Image(i)
                                                     for i in reconstr_imgs]
                if ui:
                    ui.update_image(original_imgs[0],
                                    reconstr_imgs[0])

            # for metric_name, metric_fn in train_metrics.items():
            #     log_dict[metric_name] = metric_fn(output, target)

            # Update logging and progress bar
            p_bar.set_description(f"Epoch: {epoch + 1} of {epochs} | "
                                  f"Training loss: {joint_loss.data:.5f}")
            p_bar.update(1)

            logging.log(log_dict, step=step)
            if ui:
                current_time = perf_counter()
                rate = 1 / (current_time - start_time)
                start_time = current_time
                ui.update_data(batch_idx + 1, epoch + 1, elbo_loss,
                               class_loss, rate)

        # Validation
        model.eval()
        if ui:
            ui.update_status("Running validation...")
            start_time = perf_counter()
        original_xs = []
        reconstr_xs = []
        all_mus = []
        all_log_sigmas = []
        all_targets = []
        all_preds = []

        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(device), target.to(device)

            # Forward pass
            reconstr_x, class_preds, mus, log_sigmas = model(data)

            # Add stuff to total list
            original_xs.append(data.detach())
            reconstr_xs.append(reconstr_x.detach())
            all_mus.append(mus.detach())
            all_log_sigmas.append(log_sigmas.detach())
            all_targets.append(target)
            all_preds.append(class_preds)

            log_dict = {
                "valid_mus": wandb.Histogram(mus.mean(dim=0)
                                             .detach()
                                             .cpu()),
                "valid_log_sigmas": wandb.Histogram(log_sigmas.mean(dim=0)
                                                    .detach()
                                                    .cpu())
            }

            if batch_idx % 20 == 0:
                original_imgs = tensor_to_image(data, (0, 3))
                reconstr_imgs = tensor_to_image(reconstr_x, (0, 3))

                log_dict["valid_inputs"] = [wandb.Image(i)
                                            for i in original_imgs]
                log_dict["valid_reconstructions"] = [wandb.Image(i)
                                                      for i in reconstr_imgs]

            # Calculate the metrics if this is the last batch
            if batch_idx == len(valid_loader) - 1:
                # Concat all necessary data
                x = torch.concat(original_xs, dim=0)
                x_reconstr = torch.concat(reconstr_xs, dim=0)
                mu = torch.concat(all_mus, dim=0)
                log_sigma = torch.concat(all_log_sigmas, dim=0)
                target = torch.concat(all_targets, dim=0)
                class_preds = torch.concat(all_preds, dim=0)

                # Calculate metrics
                ood_metric = ood_prediction_metric(
                    x, x_reconstr, mu, log_sigma, target
                )
                mask = target < 5
                cross_entropy = ce_metric(class_preds[mask], target[mask])

                log_dict["valid_ood"] = ood_metric.data
                log_dict["valid_class_loss"] = cross_entropy.data

            # Log metrics
            logging.log(log_dict, step=step)

        if ui:
            current_time = perf_counter()
            rate = 1 / (current_time - start_time)
            start_time = current_time
            ui.update_data(batch_idx + 1, epoch + 1, ood_metric,
                           cross_entropy, rate, True)

        # Save checkpoint if better than previous runs
        if ood_metric.data > prev_ood_metric:
            prev_checkpoint = (checkpoint_path / f"{logging.name}-"
                                                 f"ep-{epoch - 1}.pth")
            if prev_checkpoint.exists():
                prev_checkpoint.unlink()

            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "train_metrics": None,
                        "valid_metrics": None},
                       checkpoint_path / f"{logging.name}-ep-{epoch}.pth")
            prev_ood_metric = ood_metric
        # Cleanup
        del original_imgs
        del reconstr_imgs
        del all_mus
        del all_log_sigmas
        del all_targets
        del all_preds
        if ui:
            ui.update_image(tensor_to_image(data, (0, 2))[0],
                            tensor_to_image(reconstr_x, (0, 2))[0])


def train(encoder: nn.Module,
          null_multitask_head: nn.Module,
          embed_multitask_head: nn.Module,
          embedding_agent: any,
          train_embedding_agent: bool,
          null_embedding: torch.Tensor,
          encoder_optimizer: Optimizer,
          null_multitask_optimizer: Optimizer,
          embed_multitask_optimizer: Optimizer,
          kl_loss: nn.Module,
          null_loss: nn.Module,
          embed_loss: nn.Module,
          train_loader: DataLoader,
          valid_loader: DataLoader,
          epochs: int,
          logging_freq: int,
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
        null_loss: Loss function for the null MT head.
        embed_loss: Loss function for the embed MT head.
        train_loader: The training data loader.
        valid_loader: The validation data loader.
        epochs: The number of epochs to train for.
        logging_freq: The frequency at which to log training metrics.
        logging: The logger to use for logging.
        device: The device to use for training.
        chekckpoint_path: The path to save checkpoints to.
        ui: The UI to use to visualize training.
    """
    # In case any models haven't been sent to device yet
    encoder.to(device)
    null_multitask_head.to(device)
    embed_multitask_head.to(device)

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
        return z_pred, mu_pred, log_sigma_pred, b_null, y_pred_null, b_embed, \
            y_pred_embed


    step = -1

    for epoch in range(epochs):
        encoder.train()
        null_multitask_head.train()
        embed_multitask_head.train()
        embedding_agent.train()

        for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
            step += 1
            start_time = perf_counter()
            gesture = info["gesture"].to(device)

            z, mu, log_sigma, bvp_null, gesture_null, bvp_embed, \
            gesture_embed = forward_pass(amp, phase, bvp)

            # Calculate losses
            kl_loss_value = kl_loss(mu, log_sigma)
            null_loss_value = null_loss(bvp, gesture, bvp_null,
                                        gesture_null)
            embed_loss_value = embed_loss(bvp, gesture, bvp_embed,
                                          gesture_embed)

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
                "train_mus": wandb.Histogram(mu.mean(dim=0).detach().cpu()),
                "train_log_sigmas": wandb.Histogram(log_sigma.mean(dim=0)
                                                    .detach()
                                                    .cpu())
            }

            # Add images every 50 batches.
            if batch_idx % 50 == 0:
                original_imgs = tensor_to_image(bvp, (0, 3))
                null_reconstr_imgs = tensor_to_image(bvp_null, (0, 3))
                embed_reconstr_imgs = tensor_to_image(bvp_embed, (0, 3))

                log_dict["train_bvp"] = [wandb.Image(i)
                                         for i in original_imgs]
                log_dict["train_bvp_null"] = [wandb.Image(i)
                                              for i in null_reconstr_imgs]
                log_dict["train_bvp_embed"] = [wandb.Image(i)
                                               for i in embed_reconstr_imgs]

                ui.update_image(original_imgs[0],
                                null_reconstr_imgs[0],
                                embed_reconstr_imgs[0])

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
            ui.step(1)

        # Freeze the models
        encoder.eval()
        null_multitask_head.eval()
        embed_multitask_head.eval()
        # Agent training
        if train_embedding_agent:
            ui.update_status("Training embedding agent...")
            for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
                start_time = perf_counter()
                gesture = info["gesture"].to(device)

                z, mu, log_sigma, bvp_null, gesture_null, bvp_embed, \
                    gesture_embed = forward_pass(amp, phase, bvp)

                # TODO do training of agent

        # Validation
        ui.update_status("Running validation...")
        # TODO


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
        .reshape([-1, 32, 32])
    img_array = (img_array * 255).astype('uint8')
    img_arrays = [Image.fromarray(img) for img in img_array]

    return img_arrays


