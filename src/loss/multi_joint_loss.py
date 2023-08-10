"""Triple Loss.

Loss function combining KL Divergence, Reconstruction, and Classification loss.

Author:
    Yvan Satyawan
"""
import math

import torch
import torch.nn as nn


class MultiJointLoss(nn.Module):
    def __init__(self, alpha: float):
        """Joint ELBO and Classification Loss for two heads.

        The ratio between reconstruction and classification is defined by
        parameter alpha and the ratio between each head is defined by beta.

        The output of the embed losses are NoneType if no embed values are
        given.

        Args:
            alpha: The ratio between reconstruction and classification loss.
                The final loss is calculated as:
                (alpha) * elbo + (1 - alpha) * class_loss.
        """
        super().__init__()

        self.l1loss = nn.L1Loss(reduction='mean')
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.neg_alpha = 1. - alpha
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
                embed_img: torch.Tensor | None,
                embed_label: torch.Tensor | None,
                mus: torch.Tensor,
                log_sigmas: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates ELBO loss based on Practical 5.2 of 2AMM10 Deep Learning.
        """
        # First calculate reconstruction loss
        null_reconstr_loss = self.l1loss(target_img, null_img)
        if embed_img is not None:
            embed_reconstr_loss = self.l1loss(target_img, embed_img)
            reconstr_loss = (null_reconstr_loss + embed_reconstr_loss) * 0.5
        else:
            embed_reconstr_loss = None
            reconstr_loss = null_reconstr_loss

        # Reshape mus and log_sigmas to be flat
        mus = mus.reshape(-1)
        log_sigmas = log_sigmas.reshape(-1)
        a = 2 * log_sigmas

        kl_loss = (0.5 * torch.mean((mus * mus) + a.exp() - a - 1))

        # Given reconstruction and KL, calculate elbo loss
        null_elbo_loss = null_reconstr_loss + kl_loss
        if embed_img is not None:
            embed_elbo_loss = embed_reconstr_loss + kl_loss
            elbo_loss = reconstr_loss + kl_loss
        else:
            embed_elbo_loss = None
            elbo_loss = null_elbo_loss

        # Cross entropy loss on labels
        null_class_loss = self.ce(null_label, target_label)
        if embed_label is not None:
            embed_class_loss = self.ce(embed_label, target_label)
            class_loss = (null_class_loss + embed_class_loss) * 0.5
        else:
            embed_class_loss = None
            class_loss = null_class_loss

        # Calculate joint losses
        null_joint_loss = self.proportional(null_elbo_loss,
                                            null_class_loss,
                                            self.alpha, self.neg_alpha)
        if embed_img is not None:
            embed_joint_loss = self.proportional(embed_elbo_loss,
                                                 embed_class_loss,
                                                 self.alpha, self.neg_alpha)
            joint_loss = (null_joint_loss + embed_joint_loss) * 0.5
        else:
            embed_joint_loss = None
            joint_loss = null_joint_loss

        return {"elbo_loss": elbo_loss,
                "class_loss": class_loss,
                "joint_loss": joint_loss,
                "null_joint_loss": null_joint_loss,
                "embed_joint_loss": embed_joint_loss}
