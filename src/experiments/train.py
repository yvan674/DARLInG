"""Train.

Training function for DARLInG.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from time import perf_counter

import matplotlib.pyplot as plt
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
from loss.triple_loss import TripleLoss


class Training:
    def __init__(self,
                 bvp_pipeline: bool,
                 encoder: nn.Module,
                 null_head: nn.Module,
                 embed_head: nn.Module,
                 embedding_agent: BaseEmbeddingAgent,
                 null_agent: BaseEmbeddingAgent,
                 encoder_optimizer: Optimizer,
                 null_head_optimizer: Optimizer,
                 embed_head_optimizer: Optimizer,
                 loss_func: TripleLoss,
                 logging: Run,
                 checkpoint_dir: Path,
                 ui: BaseUI,
                 num_classes: int = 6):
        """Performs training on DARLInG.

        Args:
            bvp_pipeline: Whether the signal preprocessing part should be
                replaced with the precalculated BVPs.
            encoder: The CNN encoder which encodes the imaged time series
                data.
            null_head: The MT head which receives the null domain embedding.
            embed_head: The MT head which recieves some non-null domain
                embedding.
            embedding_agent: The agent which performs the domain embedding.
            null_agent: The agent providing the null embedding.
            encoder_optimizer: Optimizer for the CNN encoder.
            null_head_optimizer: Optimizer for the null domain MT head.
            embed_head_optimizer: Optimizer for the non-null domain MT head.
            loss_func: The ELBO classification loss object.
            logging: The logger to use for logging.
            checkpoint_dir: The directory to save checkpoints to.
            ui: The UI to use to visualize training.
        """
        self.bvp_pipeline = bvp_pipeline
        self.encoder = encoder
        self.null_head = null_head
        self.embed_head = embed_head
        self.embedding_agent = embedding_agent
        self.null_agent = null_agent
        self.encoder_optimizer = encoder_optimizer
        self.null_head_optimizer = null_head_optimizer
        self.embed_head_optimizer = embed_head_optimizer
        self.loss_func = loss_func
        self.logging = logging
        self.checkpoint_dir = checkpoint_dir
        self.ui = ui
        self.num_classes = num_classes

        # Keep track of some statistics to ensure only the best checkpoint is
        # saved
        self.best_joint_loss = 1.0e8
        self.prev_checkpoint_fp: Path | None = None

        # Metric Objects
        self.acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.prec = Precision(task="multiclass", num_classes=num_classes)
        self.f1 = F1Score(task="multiclass", num_classes=num_classes)
        self.conf_mat = ConfusionMatrix(task="multiclass",
                                        num_classes=num_classes)

        self.step = -1

    def _forward_pass(self,
                      amp: torch.Tensor | None,
                      phase: torch.Tensor | None,
                      bvp: torch.Tensor,
                      gesture: torch.Tensor,
                      info: list[dict[str, any]],
                      device: torch.device,
                      reconstruction_loss_only: bool,
                      no_kl_loss: bool):
        """Runs a single forward pass of the entire network.

        Args:
            amp: Amplitude shift component of the CSI.
            phase: Phase shift component of the CSI.
            bvp: Ground truth BVP from the dataset.
            gesture: Gesture ground-truths for a given batch.
        """
        gesture = gesture.to(device)
        if self.bvp_pipeline:
            bvp = bvp.to(device, dtype=torch.float32)
        else:
            amp = amp.to(device, dtype=torch.float32)
            phase = phase.to(device, dytpe=torch.float32)
            bvp = bvp.to(device, dtype=torch.float32)

        # Forward pass
        z, mu, log_sigma = self.encoder(amp, phase, bvp)

        # Generate domain embeddings
        domain_embedding = self.embedding_agent(z, info)
        null_embedding = self.null_agent(z, info)
        batch_null_embedding = torch.cat(len(amp) * [self.null_embedding])

        # Run the heads
        bvp_null, gesture_null = self.null_head(
            torch.cat([z, batch_null_embedding], dim=1)
        )
        bvp_embed, gesture_embed = self.embed_head(
            torch.cat([z, domain_embedding], dim=1)
        )

        # Calculate losses
        kl_loss, null_loss, embed_loss = self.loss_func(
            bvp, gesture, mu, log_sigma,
            bvp_null, gesture_null, bvp_embed, gesture_embed,
            reconstruction_loss_only, no_kl_loss
        )

        return {"z": z,
                "mu": mu,
                "log_sigma": log_sigma,
                "bvp_null": bvp_null,
                "gesture_null": gesture_null,
                "bvp_embed": bvp_embed,
                "gesture_embed": gesture_embed,
                "kl_loss": kl_loss,
                "null_loss": null_loss,
                "embed_loss": embed_loss}

    def _train_vae(self, train_loader: DataLoader, device: torch.device,
                   epoch: int):
        """Trains just the VAE component of the model."""
        self.ui.update_status("Training VAE model...")
        for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
            self.step += 1
            start_time = perf_counter()
            pass_result = self._forward_pass(amp, phase, bvp, info["gesture"],
                                             info, device,
                                             reconstruction_loss_only=False,
                                             no_kl_loss=False)

            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.null_head_optimizer.zero_grad()
            self.embed_head_optimizer.zero_grad()
            pass_result["kl_loss"].backward()
            pass_result["null_loss"].backward()
            pass_result["embed_loss"].backward()
            self.encoder_optimizer.step()
            self.null_head_optimizer.step()
            self.embed_head_optimizer.step()

            # Calculate metrics
            kl_loss_value = pass_result["kl_loss"].data
            null_loss_value = pass_result["null_loss"].data
            embed_loss_value = pass_result["embed_loss"].data
            joint_loss_value = (kl_loss_value
                                + null_loss_value
                                + embed_loss_value)
            loss_diff = embed_loss_value - null_loss_value

            log_dict = {
                "train_loss": joint_loss_value,
                "train_kl_loss": kl_loss_value,
                "train_null_loss": null_loss_value,
                "train_embed_loss": embed_loss_value,
                "train_loss_diff": loss_diff,
                "train_mus": wandb.Histogram(pass_result["mu"].mean(dim=0)
                                             .detach()
                                             .cpu()),
                "train_log_sigmas": wandb.Histogram(
                    pass_result["log_sigma"].mean(dim=0)
                    .detach()
                    .cpu()
                )
            }

            # Add images every 50 batches.
            if batch_idx % 50 == 0:
                log_dict = self._visualize_and_set_images(
                    bvp, pass_result["bvp_null"], pass_result["bvp_embed"],
                    log_dict, "train"
                )

            current_time = perf_counter()
            rate = 1 / (current_time - start_time)

            self.ui.update_data(
                {"train_loss": joint_loss_value,
                 "train_kl_loss": kl_loss_value,
                 "train_null_loss": null_loss_value,
                 "train_embed_loss": embed_loss_value,
                 "loss_diff": loss_diff,
                 "epoch": epoch,
                 "batch": batch_idx,
                 "rate": rate}
            )
            self.logging.log(log_dict, self.step)
            self.ui.step(1)

    def _train_agent(self, train_loader: DataLoader, device: torch.device,
                     epoch: int):
        """Trains only the embedding agent."""
        self.ui.update_status("Training embedding agent...")
        for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
            self.step += 1
            start_time = perf_counter()

            pass_result = self._forward_pass(amp, phase, bvp, info["gesture"],
                                             info, device,
                                             reconstruction_loss_only=False,
                                             no_kl_loss=True)
            loss_diff = (pass_result["embed_loss"].data
                         - pass_result["null_loss"].data)
            log_dict = {"train_loss_diff": loss_diff}
            self.embedding_agent.process_reward(pass_result["z"],
                                                loss_diff)
            current_time = perf_counter()
            self.logging.log(log_dict, self.step)
            self.ui.update_data({"loss_diff": loss_diff,
                                 "rate": 1 / (current_time - start_time),
                                 "epoch": epoch})

    def _validate_holistic(self, valid_loader: DataLoader, device,
                           epoch: int) -> float:
        """Performs validation on the entire model.

        Returns:
            The average joint loss over the validation run.
        """
        self.ui.update_status("Running validation...")
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

        for batch_idx, (amp, phase, bvp, info) in enumerate(valid_loader):
            start_time = perf_counter()
            pass_result = self._forward_pass(amp, phase, bvp, info["gesture"],
                                             info, device,
                                             reconstruction_loss_only=True,
                                             no_kl_loss=False)

            # Extract data only from the losses
            kl_loss_value, null_loss_value, embed_loss_value = [
                i.data
                for i in (pass_result["kl_loss"], pass_result["null_loss"],
                          pass_result["embed_loss"])
            ]
            joint_loss_value = (kl_loss_value
                                + null_loss_value
                                + embed_loss_value)

            # Add stuff to lists
            kl_losses.append(kl_loss_value)
            joint_losses.append(joint_loss_value)
            bvp_null_losses.append(null_loss_value)
            bvp_embed_losses.append(embed_loss_value)
            gesture_gts.append(info["gesture"].detach())
            gesture_null_preds.append(pass_result["gesture_null"])
            gesture_embed_preds.append(pass_result["gesture_embed"])

            current_time = perf_counter()
            rate = 1 / (current_time - start_time)
            self.ui.update_data(
                {"valid_loss": joint_loss_value,
                 "loss_diff": embed_loss_value - null_loss_value,
                 "epoch": epoch,
                 "batch": batch_idx,
                 "rate": rate}
            )
            bvp = bvp
            bvp_null = pass_result["bvp_null"]
            bvp_embed = pass_result["bvp_embed"]

        gesture_gts = torch.tensor(gesture_gts)
        gesture_null_preds = torch.tensor(gesture_null_preds)
        gesture_embed_preds = torch.tensor(gesture_embed_preds)

        # Calculate metrics over entire validation set
        joint_losses = float(np.mean(np.array(joint_losses)))
        log_dict = {
            "valid_kl_loss": np.mean(np.array(kl_losses)),
            "valid_joint_loss": joint_losses,
            "valid_bvp_null_loss": np.mean(np.array(bvp_null_losses)),
            "valid_bvp_embed_loss": np.mean(np.array(bvp_embed_losses)),
            "valid_null_acc": self.acc(gesture_gts, gesture_null_preds),
            "valid_null_prec": self.prec(gesture_gts, gesture_null_preds),
            "valid_null_f1": self.f1(gesture_gts, gesture_null_preds),
            "valid_null_conf_mat": self._conf_matrix(gesture_gts,
                                                     gesture_null_preds),
            "valid_embed_acc": self.acc(gesture_gts, gesture_embed_preds),
            "valid_embed_prec": self.prec(gesture_gts, gesture_embed_preds),
            "valid_embed_f1": self.f1(gesture_gts, gesture_embed_preds),
            "valid_embed_conf_mat": self._conf_matrix(gesture_gts,
                                                      gesture_embed_preds)
        }

        log_dict = self._visualize_and_set_images(bvp,
                                                  bvp_null,
                                                  bvp_embed,
                                                  log_dict,
                                                  "valid")

        self.logging.log(log_dict, step=self.step)

        del kl_losses
        del bvp_null_losses
        del bvp_embed_losses
        del gesture_gts
        del gesture_null_preds
        del gesture_embed_preds
        del bvp
        del bvp_null
        del bvp_embed

        return joint_losses

    @staticmethod
    def _tensor_to_image(data: torch.Tensor, img_idxs: tuple) -> list:
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

    def _visualize_and_set_images(self,
                                  bvp: torch.Tensor,
                                  bvp_null: torch.Tensor,
                                  bvp_embed: torch.Tensor,
                                  log_dict: dict[str, any],
                                  log_prefix: str) -> dict[str, any]:
        """Visualize BVPs and puts it in the log_dict and on WandB.

        Args:
            bvp: Ground truth BVP tensor.
            bvp_null: BVP predicted by null MT head.
            bvp_embed: BVP predicted by embed MT head.
            log_dict: The log dictionary currently being used
            log_prefix: Entry prefix to add in the log dict. Usually `train` or
                `valid`.
        """
        original_imgs = self._tensor_to_image(bvp, (0, 3))
        null_reconstr_imgs = self._tensor_to_image(bvp_null, (0, 3))
        embed_reconstr_imgs = self._tensor_to_image(bvp_embed, (0, 3))

        log_dict[f"{log_prefix}_bvp"] = [wandb.Image(i)
                                         for i in original_imgs]
        log_dict[f"{log_prefix}_bvp_null"] = [wandb.Image(i)
                                              for i in null_reconstr_imgs]
        log_dict[f"{log_prefix}_bvp_embed"] = [wandb.Image(i)
                                               for i in embed_reconstr_imgs]

        self.ui.update_image(original_imgs[0],
                             null_reconstr_imgs[0],
                             embed_reconstr_imgs[0])

        return log_dict

    def _conf_matrix(self, gesture_gt, gesture_pred):
        """Generates a conf-matrix plot"""
        fig, ax = plt.subplot()
        ax.matshow(self.conf_mat(gesture_gt, gesture_pred))
        # Put VAE to eval mode
        self.encoder.eval()
        self.null_head.eval()
        self.embed_head.eval()

        return fig

    def _save_checkpoint(self, curr_joint_loss: float, epoch: int):
        """Saves a checkpoint if validation shows performance improvement."""
        if curr_joint_loss < self.best_joint_loss:
            checkpoint_fp = (self.checkpoint_dir
                             / f"{self.logging.name}-ep-{epoch}.pth")
            if self.prev_checkpoint_fp.exists():
                self.prev_checkpoint_fp.unlink()

            torch.save(
                {
                    "encoder_state_dict":
                        self.encoder.state_dict(),
                    "null_mt_head_state_dict":
                        self.null_head.state_dict(),
                    "embed_mt_head_state_dict":
                        self.embed_head.state_dict(),
                    "embed_agent_state_dict":
                        self.embedding_agent.state_dict()
                },
                checkpoint_fp
            )
            self.best_joint_loss = curr_joint_loss
            self.prev_checkpoint_fp = checkpoint_fp

    def train(self,
              train_embedding_agent: bool,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int,
              device: torch.device):
        """Trains the model which was given in the initialization of the class.

        Args:
            train_embedding_agent: Whether to train the embedding agent or not.
            train_loader: DataLoader containing the training data.
            valid_loader: DataLoader containing the validation data.
            epochs: Number of epochs to train for
            device: Which device to train on.
        """
        # Ensure that the models are on the right devices
        self.encoder.to(device)
        self.null_head.to(device)
        self.embed_head.to(device)
        self.embedding_agent.to(device)

        for epoch in range(epochs):
            # Train the VAE portion of the model
            self.encoder.train()
            self.null_head.train()
            self.embed_head.train()
            self.embedding_agent.eval()

            self._train_vae(train_loader, device, epoch)

            # Put VAE to eval mode
            self.encoder.eval()
            self.null_head.eval()
            self.embed_head.eval()

            # Train the embedding agent if desired
            if train_embedding_agent:
                self.embedding_agent.train()

                self._train_agent(train_loader, device, epoch)

                self.embedding_agent.eval()

            # Perform validation
            curr_joint_loss = self._validate_holistic(
                valid_loader, device, epoch
            )

            # Save checkpoint
            self._save_checkpoint(curr_joint_loss, epoch)
