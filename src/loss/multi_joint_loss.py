"""Triple Loss.

Loss function combining KL Divergence, Reconstruction, and Classification loss.

Author:
    Yvan Satyawan
"""
import math

import torch
import torch.nn as nn


class MultiJointLoss(nn.Module):
    def __init__(self, alpha: float, beta: float):
        """Joint ELBO and Classification Loss for two heads.

        The ratio between reconstruction and classification is defined by
        parameter alpha and the ratio between each head is defined by beta.

        Args:
            alpha: The ratio between reconstruction and classification loss.
                The final loss is calculated as:
                (alpha) * elbo + (1 - alpha) * class_loss.
            beta: The ratio between null and embed head losses.
                The final ratio is calculated as:
                (beta) * null_loss, (1-beta) * embed_loss
        """
        super().__init__()

        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.neg_alpha = 1. - alpha
        self.beta = beta
        self.neg_beta = 1. - beta
        self.const_term = 0.5 * math.log(math.pi)

    @staticmethod
    def proportional(pos_value: torch.Tensor,
                     neg_value: torch.Tensor,
                     val: float, neg_val: float) -> torch.Tensor:
        """Combines the values using beta to provide proper weighting."""
        return (val * pos_value) + (neg_val * neg_value)

    def forward(self,
                target_img: torch.Tensor,
                target_label: torch.Tensor,
                null_img: torch.Tensor,
                null_label: torch.Tensor,
                embed_img: torch.Tensor,
                embed_label: torch.Tensor,
                mus: torch.Tensor,
                log_sigmas: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates ELBO loss based on Practical 5.2 of 2AMM10 Deep Learning.
        """
        # First calculate reconstruction loss
        null_reconstr_loss = self.mse(target_img, null_img)
        embed_reconstr_loss = self.mse(target_img, embed_img)
        reconstr_loss = self.proportional(null_reconstr_loss,
                                          embed_reconstr_loss,
                                          self.beta, self.neg_beta)

        # Reshape mus and log_sigmas to be flat
        mus = mus.reshape(-1)
        log_sigmas = log_sigmas.reshape(-1)
        a = 2 * log_sigmas

        kl_loss = (0.5 * torch.mean((mus * mus) + a.exp() - a - 1))

        # Given reconstruction and KL, calculate elbo loss
        null_elbo_loss = null_reconstr_loss + kl_loss
        embed_elbo_loss = embed_reconstr_loss + kl_loss
        elbo_loss = reconstr_loss + kl_loss

        # Cross entropy loss on labels
        null_class_loss = self.ce(null_label, target_label)
        embed_class_loss = self.ce(embed_label, target_label)
        class_loss = self.proportional(null_class_loss,
                                       embed_class_loss,
                                       self.beta, self.neg_beta)

        # Calculate joint losses
        null_joint_loss = self.proportional(null_elbo_loss,
                                            null_class_loss,
                                            self.alpha, self.neg_alpha)
        embed_joint_loss = self.proportional(embed_elbo_loss,
                                             embed_class_loss,
                                             self.alpha, self.neg_alpha)
        joint_loss = self.proportional(elbo_loss,
                                       class_loss,
                                       self.alpha, self.neg_alpha)


        return {"elbo_loss": elbo_loss,
                "class_loss": class_loss,
                "joint_loss": joint_loss,
                "null_joint_loss": null_joint_loss,
                "embed_joint_loss": embed_joint_loss}
