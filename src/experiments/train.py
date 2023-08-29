"""Train.

Training function for DARLInG.
"""
import copy
import math
from pathlib import Path

import numpy as np
from time import perf_counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy, Precision, F1Score, \
    ConfusionMatrix
import wandb
from wandb.wandb_run import Run

from models.known_domain_agent import KnownDomainAgent
from models.multi_task import run_heads
from models.null_agent import NullAgent
from loss.multi_joint_loss import MultiJointLoss
from rl.latent_environment import LatentEnvironment
from ui.base_ui import BaseUI
from utils.colors import colorcet_to_image_palette
from utils.images import tensor_to_image


class Training:
    def __init__(self,
                 bvp_pipeline: bool,
                 encoder: nn.Module,
                 null_head: nn.Module,
                 embedding_agent: str,
                 null_agent: NullAgent,
                 encoder_optimizer: Optimizer,
                 null_head_optimizer: Optimizer,
                 loss_func: MultiJointLoss,
                 logging: Run,
                 checkpoint_dir: Path,
                 ui: BaseUI,
                 embed_head_optim_class: callable,
                 lr: float,
                 reward_function: callable,
                 num_classes: int = 6,
                 agent_start_epoch: int = 0):
        """Performs training on DARLInG.

        Args:
            bvp_pipeline: Whether the signal preprocessing part should be
                replaced with the precalculated BVPs.
            encoder: The CNN encoder which encodes the imaged time series
                data.
            null_head: The MT head which receives the null domain embedding.
            embedding_agent: The agent which performs the domain embedding.
            null_agent: The agent providing the null embedding.
            encoder_optimizer: Optimizer for the CNN encoder.
            null_head_optimizer: Optimizer for the null domain MT head.
            loss_func: The ELBO classification loss object.
            logging: The logger to use for logging.
            checkpoint_dir: The directory to save checkpoints to.
            ui: The UI to use to visualize training.
            embed_head_optim_class: The class of the embed head optimizer.
            lr: The learning rate to use.
            reward_function: The reward function to use.
            num_classes: The number of classes to predict
            agent_start_epoch: Which epoch to start training the agent.
        """
        self.bvp_pipeline = bvp_pipeline
        self.encoder = encoder
        self.null_head = null_head
        self.embed_head = None
        self.embedding_agent: str | BaseAlgorithm | KnownDomainAgent = (
            embedding_agent
        )
        self.null_agent = null_agent
        self.encoder_optimizer = encoder_optimizer
        self.null_head_optimizer = null_head_optimizer
        self.embed_head_optimizer = None
        self.loss_func = loss_func
        self.logging = logging
        self.checkpoint_dir = checkpoint_dir
        self.ui = ui
        self.embed_head_optim_class = embed_head_optim_class
        self.lr = lr
        self.reward_function = reward_function
        self.num_classes = num_classes
        self.agent_start_epoch = agent_start_epoch

        self.env = None
        self.valid_env = None

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

        self.color_palette = colorcet_to_image_palette("bgy")

    def _forward_pass(self,
                      amp: torch.Tensor | None,
                      phase: torch.Tensor | None,
                      bvp: torch.Tensor,
                      gesture: torch.Tensor,
                      info: list[dict[str, any]],
                      device: torch.device,
                      no_grad_vae: bool,
                      no_grad_agent: bool) -> dict[str, torch.Tensor]:
        """Runs a single forward pass of the entire network.

        Args:
            amp: Amplitude shift component of the CSI.
            phase: Phase shift component of the CSI.
            bvp: Ground truth BVP from the dataset.
            gesture: Gesture ground-truths for a given batch.
            info: Info dictionary from the dataset.
            device: Device to train on.
            no_grad_vae: Applyl no grad to the VAE
        """
        gesture = gesture.to(device)
        if self.bvp_pipeline:
            bvp = bvp.to(device)
        else:
            amp, phase, bvp = amp.to(device), phase.to(device), bvp.to(device)

        # Forward pass
        if no_grad_vae:
            with torch.no_grad():
                z, mu, log_sigma = self.encoder(amp, phase, bvp)
        else:
            z, mu, log_sigma = self.encoder(amp, phase, bvp)

        # Generate domain embeddings
        null_embedding = self.null_agent(z, info)
        if self.embed_head is not None:
            if isinstance(self.embedding_agent, BaseAlgorithm):
                observation = z.detach().cpu()
                action, _ = self.embedding_agent.predict(observation)
                domain_embedding = torch.tensor(action, device=device,
                                                dtype=torch.float32)
            elif isinstance(self.embedding_agent, KnownDomainAgent):
                action = self.embedding_agent.predict(info)
                domain_embedding = action
            else:
                raise ValueError("How did we get here?")
            out_dict = {"agent_action": action}
        else:
            out_dict = {}
            domain_embedding = None

        # Run the heads
        if no_grad_vae:
            with torch.no_grad():
                bvp_null, gesture_null, bvp_embed, gesture_embed = run_heads(
                    self.null_head, self.embed_head,
                    null_embedding, domain_embedding, z
                )
        else:
            bvp_null, gesture_null, bvp_embed, gesture_embed = run_heads(
                self.null_head, self.embed_head,
                null_embedding, domain_embedding, z
            )

        loss_dict = self.loss_func(
            bvp, gesture,
            bvp_null, gesture_null,
            bvp_embed, gesture_embed,
            mu, log_sigma
        )
        out_dict.update({
            # Encoder
            "z": z,
            "mu": mu,
            "log_sigma": log_sigma,
            # Decoders
            "bvp_null": bvp_null,
            "gesture_null": gesture_null,
            "bvp_embed": bvp_embed,
            "gesture_embed": gesture_embed,
            # Losses
            "elbo_loss": loss_dict["elbo_loss"],
            "class_loss": loss_dict["class_loss"],
            "null_loss": loss_dict["null_joint_loss"],
            "embed_loss": loss_dict["embed_joint_loss"],
            "joint_loss": loss_dict["joint_loss"]
        })
        return out_dict

    def _train_vae(self, train_loader: DataLoader, device: torch.device,
                   epoch: int) -> bool:
        """Trains just the VAE component of the model.

        Returns:
            True if the model training was successful, False if loss goes to
            nan.
        """
        self.ui.update_status("Training VAE model...")
        for batch_idx, (amp, phase, bvp, info) in enumerate(train_loader):
            self.step += 1
            pass_result = self._forward_pass(amp, phase, bvp, info["gesture"],
                                             info, device,
                                             no_grad_vae=False,
                                             no_grad_agent=True)

            # Backward pass
            self.encoder_optimizer.zero_grad()
            self.null_head_optimizer.zero_grad()
            if self.embed_head_optimizer is not None:
                self.embed_head_optimizer.zero_grad()
            pass_result["joint_loss"].backward()
            self.encoder_optimizer.step()
            self.null_head_optimizer.step()
            if self.embed_head_optimizer is not None:
                self.embed_head_optimizer.step()

            # Calculate metrics
            elbo_loss_value = pass_result["elbo_loss"].item()
            null_loss_value = pass_result["null_loss"].item()
            joint_loss_value = (elbo_loss_value
                                + null_loss_value)
            # Check if any loss values are nan
            should_exit = (np.isnan(elbo_loss_value)
                           or np.isnan(null_loss_value)
                           or np.isnan(joint_loss_value))

            if pass_result["embed_loss"] is not None:
                embed_loss_value = pass_result["embed_loss"].item()
                joint_loss_value += embed_loss_value
                loss_diff = embed_loss_value - null_loss_value
                if np.isnan(embed_loss_value):
                    should_exit = True
            else:
                loss_diff = 0.0
                embed_loss_value = float("nan")

            loss_vals = {
                "train_loss": joint_loss_value,
                "train_elbo_loss": elbo_loss_value,
                "train_null_loss": null_loss_value,
                "train_embed_loss": embed_loss_value,
                "train_loss_diff": loss_diff,
            }

            log_dict = {
                "train_mus": wandb.Histogram(pass_result["mu"].mean(dim=0)
                                             .detach()
                                             .cpu()),
                "train_log_sigmas": wandb.Histogram(
                    pass_result["log_sigma"].mean(dim=0)
                    .detach()
                    .cpu()
                )
            }
            log_dict.update(**loss_vals)

            # Add images every 50 batches.
            if batch_idx % 50 == 0:
                log_dict.update(**self._visualize_and_set_images(
                    bvp, pass_result["bvp_null"], pass_result["bvp_embed"],
                    "train"
                ))

            ui_data = {"epoch": epoch, "batch": batch_idx}
            ui_data.update(**loss_vals)

            self.ui.update_data(ui_data)
            self.logging.log(log_dict, self.step)
            self.ui.step(len(info["user"]))

            if should_exit:
                self.ui.update_status("Joint loss is nan, exiting...")
                return False

        return True


    def _validate_holistic(self, valid_loader: DataLoader, device,
                           epoch: int) -> float:
        """Performs validation on the entire model.

        Returns:
            The average joint loss over the validation run.
        """
        self.ui.update_status("Running validation...")
        elbo_losses = []
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
            pass_result = self._forward_pass(amp, phase, bvp, info["gesture"],
                                             info, device,
                                             True, True)

            # Extract data only from the losses
            elbo_loss_value = pass_result["elbo_loss"].item()
            null_loss_value = pass_result["null_loss"].item()
            joint_loss_value = elbo_loss_value + null_loss_value
            # Handle embed, since it may be None.
            if pass_result["embed_loss"] is not None:
                embed_loss_value = pass_result["embed_loss"].item()
                joint_loss_value += embed_loss_value
            else:
                embed_loss_value = None
            # Add stuff to lists
            elbo_losses.append(elbo_loss_value)
            joint_losses.append(joint_loss_value)
            bvp_null_losses.append(null_loss_value)
            bvp_embed_losses.append(embed_loss_value)
            gesture_gts.append(info["gesture"].detach())
            gesture_null_preds.append(pass_result["gesture_null"])
            gesture_embed_preds.append(pass_result["gesture_embed"])

            data_dict = {"valid_loss": joint_loss_value,
                         "loss_diff": 0.,
                         "epoch": epoch,
                         "batch": batch_idx}
            if embed_loss_value is not None:
                data_dict["embed_loss"] = embed_loss_value - null_loss_value
            self.ui.update_data(data_dict)

            bvp_null = pass_result["bvp_null"]
            bvp_embed = pass_result["bvp_embed"]
            self.ui.step(len(info["user"]))

        gesture_gts = torch.cat(gesture_gts)
        gesture_null_preds = torch.cat(gesture_null_preds)
        gesture_null_preds = torch.argmax(gesture_null_preds, dim=1)

        # Move all to cpu
        gesture_gts = gesture_gts.to(torch.device("cpu"))
        gesture_null_preds = gesture_null_preds.to(torch.device("cpu"))

        if gesture_embed_preds[0] is not None:
            gesture_embed_preds = torch.cat(gesture_embed_preds)
            gesture_embed_preds = torch.argmax(gesture_embed_preds, dim=1)
            gesture_embed_preds = gesture_embed_preds.to(torch.device("cpu"))
        else:
            gesture_embed_preds = None

        # Calculate metrics over entire validation set
        joint_losses = np.mean(np.array(joint_losses)).item()
        log_dict = {
            "valid_elbo_loss": np.mean(np.array(elbo_losses)),
            "valid_joint_loss": joint_losses,
            "valid_bvp_null_loss": np.mean(np.array(bvp_null_losses)),
            "valid_bvp_embed_loss": float("nan"),
            "valid_null_acc": self.acc(gesture_gts, gesture_null_preds),
            "valid_null_prec": self.prec(gesture_gts, gesture_null_preds),
            "valid_null_f1": self.f1(gesture_gts, gesture_null_preds),
            "valid_null_conf_mat": self._conf_matrix(gesture_gts,
                                                     gesture_null_preds),
            "valid_embed_acc": float("nan"),
            "valid_embed_prec": float("nan"),
            "valid_embed_f1": float("nan"),
        }
        if gesture_embed_preds is not None:
            log_dict.update({
                "valid_bvp_embed_loss": np.mean(np.array(bvp_embed_losses)),
                "valid_embed_acc": self.acc(gesture_gts, gesture_embed_preds),
                "valid_embed_prec": self.prec(gesture_gts, gesture_embed_preds),
                "valid_embed_f1": self.f1(gesture_gts, gesture_embed_preds),
                "valid_embed_conf_mat": self._conf_matrix(gesture_gts,
                                                          gesture_embed_preds)
            })

        log_dict.update(**self._visualize_and_set_images(bvp,
                                                         bvp_null,
                                                         bvp_embed, "valid"))

        self.logging.log(log_dict, step=self.step)

        del elbo_losses
        del bvp_null_losses
        del bvp_embed_losses
        del gesture_gts
        del gesture_null_preds
        del gesture_embed_preds
        del bvp
        del bvp_null
        del bvp_embed
        plt.close("all")

        return joint_losses

    def _visualize_and_set_images(self,
                                  bvp: torch.Tensor,
                                  bvp_null: torch.Tensor,
                                  bvp_embed: torch.Tensor | None,
                                  log_prefix: str) -> dict[str, any]:
        """Visualize BVPs and puts it in the log_dict and on WandB.

        Args:
            bvp: Ground truth BVP tensor.
            bvp_null: BVP predicted by null MT head.
            bvp_embed: BVP predicted by embed MT head.
            log_prefix: Entry prefix to add in the log dict. Usually `train` or
                `valid`.
        """
        img_dict = {}
        original_imgs = tensor_to_image(bvp, (0, 3), self.color_palette)
        null_reconstr_imgs = tensor_to_image(bvp_null, (0, 3),
                                             self.color_palette)
        if bvp_embed is not None:
            embed_reconstr_imgs = tensor_to_image(bvp_embed, (0, 3),
                                                  self.color_palette)
            img_dict[f"{log_prefix}_bvp_embed"] = [wandb.Image(i)
                                                   for i in embed_reconstr_imgs]
        else:
            embed_reconstr_imgs = [None]

        img_dict[f"{log_prefix}_bvp"] = [wandb.Image(i)
                                         for i in original_imgs]
        img_dict[f"{log_prefix}_bvp_null"] = [wandb.Image(i)
                                              for i in null_reconstr_imgs]

        self.ui.update_image(original_imgs[0],
                             null_reconstr_imgs[0],
                             embed_reconstr_imgs[0])

        return img_dict

    def _conf_matrix(self, gesture_gt, gesture_pred):
        """Generates a conf-matrix plot"""
        fig, ax = plt.subplots()
        ax.matshow(self.conf_mat(gesture_gt, gesture_pred))

        return fig

    def _save_checkpoint(self, curr_joint_loss: float, epoch: int):
        """Saves a checkpoint if validation shows performance improvement."""
        if epoch < self.agent_start_epoch:
            return
        if curr_joint_loss < self.best_joint_loss:
            checkpoint_fp = (self.checkpoint_dir
                             / f"{self.logging.name}-ep-{epoch}.pth")
            agent_fp = (self.checkpoint_dir
                        / f"{self.logging.name}_agent-ep-{epoch}.zip")
            if self.prev_checkpoint_fp is not None:
                if self.prev_checkpoint_fp.exists():
                    self.prev_checkpoint_fp.unlink()
            save_dict = {
                "encoder_state_dict": self.encoder.state_dict(),
                "null_mt_head_state_dict": self.null_head.state_dict(),
            }
            if self.embed_head is not None:
                save_dict["embed_mt_head_state_dict"] = \
                    self.embed_head.state_dict()

            torch.save(save_dict, checkpoint_fp)
            self.embedding_agent.save(agent_fp)
            self.best_joint_loss = curr_joint_loss
            self.prev_checkpoint_fp = checkpoint_fp

    def train(self,
              train_embedding_agent: bool,
              train_loader: DataLoader,
              valid_loader: DataLoader,
              epochs: int,
              agent_epochs: int,
              device: torch.device) -> bool:
        """Trains the model which was given in the initialization of the class.

        Args:
            train_embedding_agent: Whether to train the embedding agent or not.
            train_loader: DataLoader containing the training data.
            valid_loader: DataLoader containing the validation data.
            epochs: Number of epochs to train for.
            agent_epochs: Number of epochs to train the embedding agent for.
            device: Which device to train on.

        Returns:
            True if training was successful or False if training loss was NaN.
        """
        # Ensure that the models are on the right devices
        self.encoder.to(device)
        self.null_head.to(device)

        total_agent_timesteps = agent_epochs

        for epoch in range(epochs):
            # Check if we should start training the agent yet
            if epoch == self.agent_start_epoch:
                # Duplicate the null head to create the embedding head
                self.ui.update_status("Training embedding agent starting "
                                      "this epoch...")
                self.embed_head = copy.deepcopy(self.null_head)
                self.embed_head_optimizer = self.embed_head_optim_class(
                    self.embed_head.parameters(),
                    lr=self.lr
                )
                self.embed_head.to(device)

                # Create the environment
                self.env = LatentEnvironment(
                    encoder=self.encoder,
                    null_head=self.null_head,
                    embed_head=self.embed_head,
                    null_agent=self.null_agent,
                    bvp_pipeline=self.bvp_pipeline,
                    device=device,
                    dataset=train_loader.dataset,
                    reward_function=self.reward_function
                )

                # Set up the agent
                if self.embedding_agent == "known":
                    self.embedding_agent = KnownDomainAgent(
                        self.embed_head.domain_label_size
                    )
                    self.embedding_agent.to(device)
                elif self.embedding_agent == "ddpg":
                    n_actions = self.env.action_space.shape[-1]
                    action_noise = NormalActionNoise(
                        mean=np.zeros(n_actions),
                        sigma=np.full(n_actions, 0.1)
                    )
                    self.embedding_agent = DDPG("MlpPolicy",
                                                self.env,
                                                action_noise=action_noise,
                                                device=device)
                elif self.embedding_agent == "ppo":
                    self.embedding_agent = PPO("MlpPolicy",
                                               self.env,
                                               device=device)
                    total_agent_timesteps *= len(train_loader.dataset)

            # Train the embedding agent if desired
            if train_embedding_agent and epoch >= self.agent_start_epoch:
                # Put VAE to eval mode
                self.encoder.eval()
                self.null_head.eval()
                if self.embed_head is not None:
                    self.embed_head.eval()

                self.ui.update_status(f"Training agent for "
                                      f"{total_agent_timesteps} steps...")

                self.embedding_agent.learn(total_agent_timesteps)

            # Train the VAE portion of the model
            self.encoder.train()
            self.null_head.train()
            if self.embed_head is not None:
                self.embed_head.train()

            if not self._train_vae(train_loader, device, epoch):
                return False

            # Put VAE to eval mode
            self.encoder.eval()
            self.null_head.eval()
            if self.embed_head is not None:
                self.embed_head.eval()

            # Perform validation
            curr_joint_loss = self._validate_holistic(
                valid_loader, device, epoch
            )
            if math.isnan(curr_joint_loss):
                self.ui.update_status("Validation loss is NaN.")
                return False

            # Save checkpoint
            self._save_checkpoint(curr_joint_loss, epoch)
        return True
