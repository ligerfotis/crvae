from torch import nn


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, color_channels, output_size):
        super().__init__()
        out_channel = color_channels
        self.decoder_lin_1 = nn.Sequential(nn.Linear(encoded_space_dim, 4 * 4 * 256), nn.ReLU(True))
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 4, 4))
        if output_size == 32:
            output_padding = [1, 1, 1]
            padding = [1, 1, 1]
        else:
            output_padding = [0, 0, 1]
            padding = [1, 1, 0]
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=padding[0],
                               output_padding=output_padding[0]),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=padding[1],
                               output_padding=output_padding[1]),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=padding[2],
                               output_padding=output_padding[2]),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channel, 3, stride=1, padding=1),
        )
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.decoder_lin_1(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = self.sigmoid(x)
        return x

import math
import torch.nn as nn


class DecoderBase(nn.Module):
    """docstring for Decoder"""

    def __init__(self):
        super(DecoderBase, self).__init__()

    def decode(self, x, z):
        raise NotImplementedError

    def reconstruct_error(self, x, z):
        """reconstruction loss
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            loss: (batch_size, n_sample). Loss
            across different sentence and z
        """

        raise NotImplementedError

    def beam_search_decode(self, z, K):
        """beam search decoding
        Args:
            z: (batch_size, nz)
            K: the beam size

        Returns: List1
            List1: the decoded word sentence list
        """

        raise NotImplementedError

    def sample_decode(self, z):
        """sampling from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """

        raise NotImplementedError

    def greedy_decode(self, z):
        """greedy decoding from z
        Args:
            z: (batch_size, nz)

        Returns: List1
            List1: the decoded word sentence list
        """

        raise NotImplementedError

    def log_probability(self, x, z):
        """
        Args:
            x: (batch_size, *)
            z: (batch_size, n_sample, nz)
        Returns:
            log_p: (batch_size, n_sample).
                log_p(x|z) across different x and z
        """

        raise NotImplementedError


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, masked_channels, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :masked_channels, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :masked_channels, kH // 2 + 1:] = 0

    def reset_parameters(self):
        n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
        self.weight.data.normal_(0, math.sqrt(2. / n))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        self.weight.data.mul_(self.mask)
        return super(MaskedConv2d, self).forward(x)


class PixelCNNBlock(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(PixelCNNBlock, self).__init__()
        self.mask_type = 'B'
        padding = kernel_size // 2
        out_channels = in_channels // 2

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            MaskedConv2d(self.mask_type, out_channels, out_channels, out_channels, kernel_size, padding=padding,
                         bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
            nn.Conv2d(out_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.activation = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        return self.activation(self.main(input) + input)


class MaskABlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, masked_channels):
        super(MaskABlock, self).__init__()
        self.mask_type = 'A'
        padding = kernel_size // 2

        self.main = nn.Sequential(
            MaskedConv2d(self.mask_type, masked_channels, in_channels, out_channels, kernel_size, padding=padding,
                         bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        m = self.main[1]
        assert isinstance(m, nn.BatchNorm2d)
        m.weight.data.fill_(1)
        m.bias.data.zero_()

    def forward(self, input):
        return self.main(input)


class PixelCNN(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, kernel_sizes, masked_channels):
        super(PixelCNN, self).__init__()
        assert num_blocks == len(kernel_sizes)
        self.blocks = []
        for i in range(num_blocks):
            if i == 0:
                block = MaskABlock(in_channels, out_channels, kernel_sizes[i], masked_channels)
            else:
                block = PixelCNNBlock(out_channels, kernel_sizes[i])
            self.blocks.append(block)

        self.main = nn.ModuleList(self.blocks)

        self.direct_connects = []
        for i in range(1, num_blocks - 1):
            self.direct_connects.append(PixelCNNBlock(out_channels, kernel_sizes[i]))

        self.direct_connects = nn.ModuleList(self.direct_connects)

    def forward(self, input):
        # [batch, out_channels, H, W]
        direct_inputs = []
        for i, layer in enumerate(self.main):
            if i > 2:
                direct_input = direct_inputs.pop(0)
                direct_conncet = self.direct_connects[i - 3]
                input = input + direct_conncet(direct_input)

            input = layer(input)
            direct_inputs.append(input)
        assert len(direct_inputs) == 3, 'architecture error: %d' % len(direct_inputs)
        direct_conncet = self.direct_connects[-1]
        return input + direct_conncet(direct_inputs.pop(0))


class PixelCNNDecoderV2(DecoderBase):
    def __init__(self, encoded_space_dim, color_channels, output_size):
        super(PixelCNNDecoderV2, self).__init__()
        self.n_latent_features = encoded_space_dim
        self.out_channel = color_channels
        self.fm_latent = 512
        self.output_size = output_size
        self.img_latent = self.output_size * self.output_size * self.fm_latent
        if self.n_latent_features != 0:
            self.z_transform = nn.Sequential(
                nn.Linear(self.n_latent_features, self.img_latent),
            )

        kernal_sizes = [7, 7, 7, 7, 7, 5, 5, 5, 5, 3, 3, 3, 3]
        kernal_sizes = kernal_sizes[:len(kernal_sizes) // 2]

        hidden_channels = 64
        self.pixel_cnn = PixelCNN(self.fm_latent, hidden_channels, len(kernal_sizes), kernal_sizes, 0)
        self.cnn_out = nn.Conv2d(hidden_channels, hidden_channels, 1, bias=False)
        self.cnn_out_bn = nn.BatchNorm2d(hidden_channels)
        self.cnn_out_act = nn.ELU()
        self.cnn_out2 = nn.Conv2d(hidden_channels, self.out_channel, 1, bias=False)
        self.cnn_out2_bn = nn.Sigmoid()

        self.reset_parameters()

    def reset_parameters(self):
        if self.n_latent_features != 0:
            nn.init.xavier_uniform_(self.z_transform[0].weight)
            nn.init.constant_(self.z_transform[0].bias, 0)
        nn.init.xavier_uniform_(self.cnn_out.weight)
        nn.init.xavier_uniform_(self.cnn_out2.weight)

    def forward(self, z):
        if self.n_latent_features != 0:
            z = self.z_transform(z)
        z = z.view(-1, self.fm_latent, self.output_size, self.output_size)
        x = self.pixel_cnn(z)
        x = self.cnn_out_act(self.cnn_out_bn(self.cnn_out(x)))
        x = self.cnn_out2_bn(self.cnn_out2(x))
        return x
