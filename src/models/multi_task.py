"""Multi Task Head.

Multitask head that predicts both the BVP and the gesture

Author:
    Yvan Satyawan <y_satyawan@hotmail.com>
    Jonas Niederle <github.com/jmniederle>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self,
                 conv_ac_func: nn.Module = nn.ReLU,
                 dropout: float = 0.3,
                 latent_dim: int = 10,
                 output_layers: int = 1,
                 output_size: tuple[int, int] | int = 20):
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
                 in_features: int = 10,
                 num_layers: int = 3):
        """Gesture Predictor using a fully connected network."""
        super().__init__()

        def linear_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(dropout),
                fc_ac_func()
            )

        output_layers = [2 ** (i + 4) for i in range(num_layers)]
        self.mlp = nn.Sequential()

        output_sizes = [(in_features, output_layers[0])] + \
                       [(output_layers[i], output_layers[i + 1])
                        for i in range(len(output_layers) - 1)] + \
                       [(output_layers[-1], num_classes)]

        for size in output_sizes:
            self.mlp.append(linear_block(size[0], size[1]))

        self.mlp.append(nn.Softmax(dim=1))

    def forward(self, x):
        return self.mlp(x)


class MultiTaskHead(nn.Module):
    def __init__(self,
                 decoder_ac_func: nn.Module,
                 decoder_dropout: float,
                 decoder_output_layers: int,
                 decoder_output_size: tuple[int, int] | int,
                 encoder_latent_dim: int,
                 predictor_num_layers: int,
                 predictor_ac_func: nn.Module,
                 predictor_dropout: float,
                 domain_label_size: int,
                 num_classes: int = 6):
        """Multi Task Prediction Head.

        Args:
            decoder_ac_func: Activation function for the decoder.
            decoder_dropout: Dropout rate for the decoder.
            decoder_output_layers: How many channels the BVP input actually has
                which we should reconstruct for.
            decoder_output_size: The size of the BVP output to reconstruct.
            encoder_latent_dim: Latent dimensionality of the output of the
                encoder.
            predictor_num_layers: Number of linear layers to use in the
                GesturePredictor.
            predictor_ac_func: Activation function for the predictor.
            predictor_dropout: Dropout rate for the predictor.
            domain_label_size: Dimensionality of the domain label.
            num_classes: Number of classes to predict.
        """
        super().__init__()

        in_features = domain_label_size + encoder_latent_dim
        self.domain_label_size = domain_label_size

        self.decoder = Decoder(decoder_ac_func,
                               decoder_dropout,
                               in_features,
                               decoder_output_layers,
                               decoder_output_size)
        self.predictor = GesturePredictor(predictor_ac_func,
                                          predictor_dropout,
                                          num_classes=num_classes,
                                          in_features=in_features,
                                          num_layers=predictor_num_layers)

    def forward(self, z):
        y_bvp = self.decoder(z)
        y_gesture = self.predictor(z)

        return y_bvp, y_gesture


def run_heads(null_head: MultiTaskHead,
              embed_head: MultiTaskHead,
              null_embedding: torch.Tensor,
              agent_embedding: torch.Tensor,
              z: torch.Tensor):
    """Runs both heads together.

    Args:
        null_head: The multitask head for the null embedding
        embed_head: The multitask head for the agent embedding
        null_embedding: The null embedding. Must be shape
            [batch_size, domain_embedding_dim].
        agent_embedding: The actual embedding. Must be shape
            [batch_size, domain_embedding_dim].
        z: The z value produced by the encoder. Must be in the shape
            [batch_size, latent_embedding_dim].

    Returns:
        bvp_null, gesture_null, bvp_embed, gesture_embed. Embed values are
        None if the embed_head is None.
    """
    bvp_null, gesture_null = null_head(
        torch.cat([z, null_embedding], dim=1)
    )
    if embed_head is not None:
        bvp_embed, gesture_embed = embed_head(
            torch.cat([z, agent_embedding], dim=1)
        )
    else:
        bvp_embed, gesture_embed = None, None

    return bvp_null, gesture_null, bvp_embed, gesture_embed
