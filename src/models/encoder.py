"""Encoder.

Based on a VAE.

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    Jonas Niederle <github.com/jmniederle>
"""
from typing import TypeVar

import torch
import torch.nn as nn


T = TypeVar('T')


class AmpPhaseEncoder(nn.Module):
    def __init__(self,
                 conv_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 fc_input_size: int = 4608,
                 input_dim: int = 18):
        """Encoder that processes the amp and the phase.

        This combines two separate encoders to encode both amp and phase images.
        """
        super().__init__()
        self.amp_encoder = Encoder(conv_ac_func, dropout, latent_dim,
                                   fc_input_size, input_dim)
        self.phase_encoder = Encoder(conv_ac_func, dropout, latent_dim,
                                     fc_input_size, input_dim)

    def forward(self, amp, phase, bvp):
        z_amp, mu_amp, log_sigma_amp = self.amp_encoder(amp)
        z_phase, mu_phase, log_sigma_phase = self.phase_encoder(phase)

        return (torch.cat((z_amp, z_phase)),
                torch.cat((mu_amp, mu_phase)),
                torch.cat((log_sigma_amp, log_sigma_phase)))


class BVPEncoder(nn.Module):
    def __init__(self,
                 conv_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 fc_input_size: int = 4608,
                 input_dim: int = 1):
        """Encoder that processes only the BVP.

        This is a wrapper of only 1 encoder such that the signature is the same
        between both the BVP only encoder and the combined amp-phase encoder.
        """
        super().__init__()
        self.encoder = Encoder(conv_ac_func, dropout, latent_dim, fc_input_size,
                               input_dim)

    def forward(self, amp, phase, bvp):
        return self.encoder(bvp)


class Encoder(nn.Module):
    def __init__(self,
                 conv_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 fc_input_size: int = 4608,
                 input_dim: int = 1):
        """Encoder model for our network.

        This is an encoder which can accept either amplitude, phase, or BVP as
        its input.
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

        self.convnet = nn.Sequential(conv_block(input_dim, 128),
                                     conv_block(128, 256),
                                     conv_block(256, 512),
                                     nn.Flatten())

        # self.fc = nn.Linear(fc_input_size, 8192)

        self.fc_mu = nn.Linear(fc_input_size, latent_dim)
        self.fc_sigma = nn.Linear(8192, latent_dim)

    def forward(self, x):
        """Forward pass. """
        h = self.convnet(x)
        # h = self.fc(h)
        mu = self.fc_mu(h)
        log_sigma = self.fc_sigma(h)
        z = self.reparameterization(mu, log_sigma)

        return z, mu, log_sigma

    @staticmethod
    def reparameterization(mu, log_sigma):
        sigma = torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
