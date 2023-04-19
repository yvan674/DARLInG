"""Encoder.

Based on a VAE.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    Jonas Niederle <github.com/jmniederle>
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 conv_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 bvp_pipeline: bool = False):
        """Encoder model for our network.

        This is an encoder which can accept either amplitude and phase as its
        input or just the BVP as its input. The number of convnet blocks (note
        not conv blocks) in the network is 2 if bvp_pipeline is False, one each
        for amp and phase. Otherwise, only one is used for the precalculated
        BVP encoding.
        """
        super(Encoder, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                conv_ac_func(),
                nn.Dropout(dropout)
            )

        if bvp_pipeline:
            self.convnets = [
                nn.Sequential(
                    conv_block(1, 128),
                    conv_block(128, 256),
                    conv_block(256, 512),
                    nn.Flatten()
                )
            ]
        else:
            self.convnets = [
                nn.Sequential(
                    conv_block(18, 128),
                    conv_block(128, 256),
                    conv_block(256, 512),
                    nn.Flatten()
                ),
                nn.Sequential(
                    conv_block(18, 128),
                    conv_block(128, 256),
                    conv_block(256, 512),
                    nn.Flatten()
                )
            ]


        self.bvp_pipeline = bvp_pipeline
        self.fc_mu_amp = nn.Linear(8192, latent_dim)
        self.fc_sigma_amp = nn.Linear(8192, latent_dim)

        self.fc_mu_phase = nn.Linear(8192, latent_dim)
        self.fc_sigma_phase = nn.Linear(8192, latent_dim)

    def forward(self, amp, phase, bvp):
        """Forward pass.

        We include the bvp parameter in this method, since we also use this
        encoder to encode the BVP data.
        """
        if self.bvp_pipeline:
            h = self.convnets[0](bvp)
        else:
            h_amp = self.convnets[0](amp)
            h_phase = self.convnets[1](phase)
            h = torch.cat([h_amp, h_phase], dim=1)


        mu = self.fc_mu(h)
        log_sigma = self.fc_sigma(h)

        z = self.reparameterization(mu, log_sigma)

        return z, mu, log_sigma

    @staticmethod
    def reparameterization(mu, log_sigma):
        sigma = torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
