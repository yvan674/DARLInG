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
                 input_dim: int = 18,
                 initial_kernel_size: int = 3,
                 conv_output_sizes: list[int] = None):
        """Encoder that processes the amp and the phase.

        This combines two separate encoders to encode both amp and phase images.
        """
        super().__init__()
        self.amp_encoder = Encoder(conv_ac_func, dropout, latent_dim,
                                   fc_input_size, input_dim,
                                   initial_kernel_size, conv_output_sizes)
        self.phase_encoder = Encoder(conv_ac_func, dropout, latent_dim,
                                     fc_input_size, input_dim,
                                     initial_kernel_size, conv_output_sizes)
        self.latent_dim = self.amp_encoder.latent_dim

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
                 input_dim: int = 1,
                 initial_kernel_size: int = 3,
                 conv_output_sizes: list[int] = None):
        """Encoder that processes only the BVP.

        This is a wrapper of only 1 encoder such that the signature is the same
        between both the BVP only encoder and the combined amp-phase encoder.
        """
        super().__init__()
        self.encoder = Encoder(conv_ac_func, dropout, latent_dim, fc_input_size,
                               input_dim, initial_kernel_size,
                               conv_output_sizes)
        self.latent_dim = self.encoder.latent_dim

    def forward(self, amp, phase, bvp):
        return self.encoder(bvp)


class Encoder(nn.Module):
    def __init__(self,
                 conv_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 fc_input_size: int = 4608,
                 input_dim: int = 1,
                 initial_kernel_size: int = 3,
                 conv_output_sizes: list[int] = None):
        """Encoder model for our network.

        This is an encoder which can accept either amplitude, phase, or BVP as
        its input.

        Args:
            conv_ac_func: The activation function to use after each convolution.
            dropout: The dropout rate to use after each convolution.
            latent_dim: The dimension of the latent space.
            fc_input_size: The size of the input to the fully connected layer.
            input_dim: The dimension of the input.
            initial_kernel_size: The size of the initial kernels. The kernel
                sizes are constant throughout the network until the final 2
                convolutions, which have a kernel size of 3.
            conv_output_sizes: The output sizes of the convolutional layers. If
                None, the default values of [128, 256, 512] are used.
        """
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        def conv_block(in_channels, out_channels, kernel_size):
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=kernel_size, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                conv_ac_func(),
                nn.Dropout(dropout)
            )

        # Set default output sizes.
        if conv_output_sizes is None:
            conv_output_sizes = [128, 256, 512]

        conv_filter_sizes = [(input_dim, conv_output_sizes[0])] + \
                            [(conv_output_sizes[i], conv_output_sizes[i + 1])
                             for i in range(len(conv_output_sizes) - 1)]

        kernel_sizes = [initial_kernel_size, initial_kernel_size] + \
                       [3 for _ in range(len(conv_output_sizes) - 2)]

        self.convnet = nn.Sequential()

        for size, kernel in zip(conv_filter_sizes, kernel_sizes):
            self.convnet.append(conv_block(size[0], size[1], kernel))

        # self.fc = nn.Linear(fc_input_size, 8192)

        self.fc_mu = nn.Sequential(
            nn.Linear(fc_input_size, 8192),
            conv_ac_func(),
            nn.Linear(8192, latent_dim),
            conv_ac_func()
        )

        self.fc_sigma = nn.Sequential(
            nn.Linear(fc_input_size, 8192),
            conv_ac_func(),
            nn.Linear(8192, latent_dim),
            conv_ac_func()
        )

    def forward(self, x):
        """Forward pass. """
        h = self.convnet(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        log_sigma = self.fc_sigma(h)
        z = self.reparameterization(mu, log_sigma)
        z = torch.sigmoid(z)

        return z, mu, log_sigma

    @staticmethod
    def reparameterization(mu, log_sigma):
        sigma = torch.exp(log_sigma)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon
