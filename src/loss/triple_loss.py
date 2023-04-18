"""Triple Loss.

Loss function combining KL Divergence, Reconstruction, and Classification loss.

Author:
    Yvan Satyawan
"""
import math

import torch
import torch.nn as nn


class TripleLoss(nn.Module):
    def __init__(self, alpha):
        """Combination of KL, Reconstruction, and Classification Loss.

        The ratio between reconstruction and classification is defined by
        parameter alpha.

        Args:
            alpha: The ratio between reconstruction and classification loss.
                The final loss is calculated as:
                (alpha) * reconstr_loss + (1 - alpha) * class_loss.
        """
        super().__init__()

        self.mse = nn.MSELoss(reduction='sum')
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.neg_alpha = 1. - alpha
        self.const_term = 0.5 * math.log(math.pi)

    def forward(self,
                bvp: torch.Tensor, gesture: torch.Tensor,
                mu: torch.Tensor, log_sigma: torch.Tensor,
                bvp_null: torch.Tensor, gesture_null: torch.Tensor,
                bvp_embed: torch.Tensor, gesture_embed: torch.Tensor,
                reconstr_loss_only: bool, no_kl_loss: bool
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates ELBO loss based on 2AMM10 Practical 5.2.

        Args:
            bvp: Ground truth BVP image.
            gesture: Ground truth gesture.
            mu: mu value calculated by the encoder.
            log_sigma: Log_sigma value calculated by the encoder.
            bvp_null: Predicted BVP image from the null model.
            gesture_null: Predicted gesture from the null model.
            bvp_embed: Predicted BVP image from the embedding model.
            gesture_embed: Predicted gesture from the embedding model.
            reconstr_loss_only: Whether to only calculate the reconstruction
                loss.
            no_kl_loss: Whether to not calculate the KL loss.

        Returns:
            - KL Divergence.
            - Reconstruction [+ classification] of the null prediction.
            - Reconstruction [+ classification] of the embedding prediction.
        """
        if no_kl_loss:
            kl_loss = None
        else:
            # Reshape mus and log_sigmas to be flat
            mus = mu.reshape(-1)
            log_sigmas = log_sigma.reshape(-1)
            a = 2 * log_sigmas
            kl_loss = (0.5 * torch.sum(mus * mus + a.exp() - a - 1))

        # Calculate the reconstruction loss using MSE
        null_loss = self.mse(bvp_null, bvp)
        embed_loss = self.mse(bvp_embed, bvp)

        if not reconstr_loss_only:
            null_loss = ((self.alpha * null_loss)
                         + (self.neg_alpha * self.ce(gesture_null, gesture)))
            embed_loss = ((self.alpha * embed_loss)
                          + (self.neg_alpha * self.ce(gesture_embed, gesture)))

        return kl_loss, null_loss, embed_loss
