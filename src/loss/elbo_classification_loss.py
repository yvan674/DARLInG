"""ELBO Classification Loss.

This loss function is taken directly from our solution of 2AMM10 Deep Learning,
Assignment 4.
"""
import torch
import torch.nn as nn
import math


class ELBOClassificationLoss(nn.Module):
    def __init__(self, alpha):
        """Joint ELBO and Classification Loss.

        The ratio between ELBO and Classification is defined by parameter alpha.

        Args:
            alpha: The ratio between ELBO and Classification loss. The final
                loss is calculated as (1 - alpha) * elbo + alpha * class loss.
        """
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.const_term = 0.5 * math.log(math.pi)

    def forward(self,
                original_imgs: torch.Tensor,
                reconstr_imgs: torch.Tensor,
                class_preds: torch.Tensor,
                class_labels: torch.Tensor,
                mus: torch.Tensor,
                log_sigmas: torch.Tensor):
        """Calculates ELBO loss based on Practical 5.2"""
        # Calculate the construction loss using MSE
        reconstr_loss = self.mse(reconstr_imgs, original_imgs)

        # Reshape mus and log_sigmas to be flat
        mus = mus.reshape(-1)
        log_sigmas = log_sigmas.reshape(-1)
        a = 2 * log_sigmas
        kl_loss = (0.5 * torch.sum(mus * mus + a.exp() - a - 1))

        # Given reconstruction and KL, calculate elbo loss
        elbo_loss = reconstr_loss + kl_loss

        # Do cross entropy loss only on images with labels
        label_mask = class_labels < 5
        class_loss = self.ce(class_preds[label_mask],
                             class_labels[label_mask])

        joint_loss = ((1 - self.alpha) * elbo_loss) + (self.alpha * class_loss)

        # Calculate joint loss
        return elbo_loss, class_loss, joint_loss

