"""Multi Task Head.

Multitask head that predicts both the BVP and the gesture

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    Jonas Niederle <github.com/jmniederle>
"""
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,
                 conv_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 output_layers: int = 1,
                 output_size: int = 20):
        """Decoder from a VAE.
        """
        super(Decoder, self).__init__()

        self.output_size = output_size

        def conv_block_decoder(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1),
                nn.BatchNorm2d(out_channels),
                conv_ac_func(),
                nn.Dropout(dropout)
            )

        self.decoder_input = nn.Linear(latent_dim, 8192)
        self.convnet_decoder = nn.Sequential(
            conv_block_decoder(512, 256),
            conv_block_decoder(256, 128),
            conv_block_decoder(128, 128),
            nn.Conv2d(128, output_layers, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.reshape((-1, 512, 4, 4))
        x_reconstr = self.convnet_decoder(result)
        # Image is now 32x32, we need to make it whatever size is required.
        x_reconstr = F.interpolate(x_reconstr, size=self.output_size)
        return x_reconstr


class GesturePredictor(nn.Module):
    def __init__(self,
                 fc_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 num_classes: int = 6,
                 in_features: int = 10):
        """Gesture Predictor using a fully connected network."""
        super().__init__()

        def linear_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(dropout),
                fc_ac_func()
            )

        self.mlp = nn.Sequential(
            linear_block(in_features, 256),
            linear_block(256, 512),
            linear_block(512, 1024),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.mlp(x)


class MultiTaskHead(nn.Module):
    def __init__(self,
                 decoder_ac_func: nn.Module,
                 decoder_dropout: float,
                 encoder_latent_dim: int,
                 predictor_ac_func: nn.Module,
                 predictor_dropout: float,
                 domain_label_size: int,
                 bvp_output_layers: int,
                 bvp_output_size: int = 20,
                 num_classes: int = 6):
        """Multi Task Prediction Head.

        Args:
            decoder_ac_func: Activation function for the decoder.
            decoder_dropout: Dropout rate for the decoder.
            encoder_latent_dim: Latent dimensionality of the output of the
                encoder.
            predictor_ac_func: Activation function for the predictor.
            predictor_dropout: Dropout rate for the predictor.
            domain_label_size: Dimensionality of the domain label.
        """
        super().__init__()

        in_features = domain_label_size + encoder_latent_dim

        self.decoder = Decoder(decoder_ac_func,
                               decoder_dropout,
                               in_features,
                               bvp_output_layers,
                               bvp_output_size)
        self.predictor = GesturePredictor(predictor_ac_func,
                                          predictor_dropout,
                                          in_features=in_features,
                                          num_classes=num_classes)

    def forward(self, z):
        y_bvp = self.decoder(z)
        y_gesture = self.predictor(z)

        return y_bvp, y_gesture
